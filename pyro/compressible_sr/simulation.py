import importlib

import matplotlib

try:
    matplotlib.rcParams['mpl_toolkits.legacy_colorbar'] = False
except KeyError:
    pass
import matplotlib.pyplot as plt
import numpy as np

import pyro.compressible_sr.BC as BC
import pyro.compressible_sr.derives as derives
import pyro.compressible_sr.eos as eos
import pyro.compressible_sr.unsplit_fluxes as flx
import pyro.mesh.boundary as bnd
import pyro.util.plot_tools as plot_tools
from pyro.simulation_null import NullSimulation, bc_setup, grid_setup

# np.seterr(all='raise')


class Variables:
    """
    a container class for easy access to the different compressible
    variable by an integer key
    """
    def __init__(self, myd):
        self.nvar = len(myd.names)

        # conserved variables -- we set these when we initialize for
        # they match the CellCenterData2d object
        try:
            self.idens = myd.names.index("density")
            self.ixmom = myd.names.index("x-momentum")
            self.iymom = myd.names.index("y-momentum")
            self.iener = myd.names.index("energy")
        except ValueError:
            self.idens = 0
            self.ixmom = 1
            self.iymom = 2
            self.iener = 3

        # if there are any additional variable, we treat them as
        # passively advected scalars
        self.naux = self.nvar - 4
        if self.naux > 0:
            self.irhox = 4
        else:
            self.irhox = -1

        # primitive variables
        self.nq = 4 + self.naux

        self.irho = 0
        self.iu = 1
        self.iv = 2
        self.ip = 3

        if self.naux > 0:
            self.ix = 4   # advected scalar
        else:
            self.ix = -1


def prim_to_cons(q, gamma, ivars, myg):
    """ convert an input vector of primitive variables to conserved variables """

    U = myg.scratch_array(nvar=ivars.nvar)

    u = q[:, :, ivars.iu]
    v = q[:, :, ivars.iv]

    try:
        W = 1 / np.sqrt(1 - u**2 - v**2)
    except FloatingPointError:
        u[np.isnan(u)] = 0
        v[np.isnan(v)] = 0
        W = np.ones_like(u)

    rhoh = eos.rhoh_from_rho_p(gamma, q[:, :, ivars.irho], q[:, :, ivars.ip])

    U[:, :, ivars.idens] = q[:, :, ivars.irho] * W
    U[:, :, ivars.ixmom] = u * rhoh * W**2
    U[:, :, ivars.iymom] = v * rhoh * W**2

    U[:, :, ivars.iener] = rhoh * W**2 - q[:, :, ivars.ip] - U[:, :, ivars.idens]

    if ivars.naux > 0:
        for nq, nu in zip(range(ivars.ix, ivars.ix+ivars.naux),
                          range(ivars.irhox, ivars.irhox+ivars.naux)):
            U[:, :, nu] = q[:, :, nq]*q[:, :, ivars.irho]*W

    return U


class Simulation(NullSimulation):
    """The main simulation class for the corner transport upwind
    compressible hydrodynamics solver

    """

    def initialize(self, extra_vars=None, ng=4):
        """
        Initialize the grid and variables for compressible flow and set
        the initial conditions for the chosen problem.
        """
        my_grid = grid_setup(self.rp, ng=ng)
        my_data = self.data_class(my_grid)

        # define solver specific boundary condition routines
        bnd.define_bc("hse", BC.user, is_solid=False)
        bnd.define_bc("ramp", BC.user, is_solid=False)  # for double mach reflection problem

        bc, bc_xodd, bc_yodd = bc_setup(self.rp)

        # are we dealing with solid boundaries? we'll use these for
        # the Riemann solver
        self.solid = bnd.bc_is_solid(bc)

        # density and energy
        my_data.register_var("density", bc)
        my_data.register_var("energy", bc)
        my_data.register_var("x-momentum", bc_xodd)
        my_data.register_var("y-momentum", bc_yodd)

        # any extras?
        if extra_vars is not None:
            for v in extra_vars:
                my_data.register_var(v, bc)

        # store the EOS gamma as an auxillary quantity so we can have a
        # self-contained object stored in output files to make plots.
        # store grav because we'll need that in some BCs
        my_data.set_aux("gamma", self.rp.get_param("eos.gamma"))
        my_data.set_aux("grav", self.rp.get_param("compressible.grav"))

        my_data.create()

        self.cc_data = my_data

        # some auxillary data that we'll need to fill GC in, but isn't
        # really part of the main solution
        aux_data = self.data_class(my_grid)
        aux_data.register_var("ymom_src", bc_yodd)
        aux_data.register_var("E_src", bc)
        aux_data.create()
        self.aux_data = aux_data

        # derived variables
        self.cc_data.add_derived(derives.derive_primitives)

        self.ivars = Variables(my_data)
        self.cc_data.add_ivars(self.ivars)

        # initial conditions for the problem
        problem = importlib.import_module("pyro.{}.problems.{}".format(
            self.solver_name, self.problem_name))
        problem.init_data(self.cc_data, self.rp)

        if self.verbose > 0:
            print(my_data)

    def method_compute_timestep(self):
        """
        The timestep function computes the advective timestep (CFL)
        constraint.  The CFL constraint says that information cannot
        propagate further than one zone per timestep.

        We use the driver.cfl parameter to control what fraction of the
        CFL step we actually take.
        """

        cfl = self.rp.get_param("driver.cfl")

        # get the variables we need
        u, v, cs = self.cc_data.get_var(["velocity", "soundspeed"])

        # print(f'u = {u}')
        # print(f'v = {v}')
        # print(f'cs = {cs}')

        # print(sum(abs(u)))

        # the timestep is min(dx/(|u| + cs), dy/(|v| + cs))
        xtmp = self.cc_data.grid.dx/(abs(u) + cs)
        ytmp = self.cc_data.grid.dy/(abs(v) + cs)

        self.dt = cfl*float(min(xtmp.min(), ytmp.min()))

    def evolve(self):
        """
        Evolve the equations of compressible hydrodynamics through a
        timestep dt.
        """

        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()

        dens = self.cc_data.get_var("density")
        ymom = self.cc_data.get_var("y-momentum")
        ener = self.cc_data.get_var("energy")

        grav = self.rp.get_param("compressible.grav")

        myg = self.cc_data.grid

        Flux_x, Flux_y = flx.unsplit_fluxes(self.cc_data, self.aux_data, self.rp,
                                            self.ivars, self.solid, self.tc, self.dt)

        old_dens = dens.copy()
        old_ymom = ymom.copy()

        # conservative update
        dtdx = self.dt/myg.dx
        dtdy = self.dt/myg.dy

        for n in range(self.ivars.nvar):
            var = self.cc_data.get_var_by_index(n)

            var.v()[:, :] += \
                dtdx*(Flux_x.v(n=n) - Flux_x.ip(1, n=n)) + \
                dtdy*(Flux_y.v(n=n) - Flux_y.jp(1, n=n))

        # gravitational source terms
        ymom[:, :] += 0.5*self.dt*(dens[:, :] + old_dens[:, :])*grav
        ener[:, :] += 0.5*self.dt*(ymom[:, :] + old_ymom[:, :])*grav

        # increment the time
        self.cc_data.t += self.dt
        self.n += 1

        tm_evolve.end()

    def dovis(self):
        """
        Do runtime visualization.
        """

        plt.clf()

        plt.rc("font", size=10)

        # we do this even though ivars is in self, so this works when
        # we are plotting from a file
        ivars = Variables(self.cc_data)

        # access gamma from the cc_data object so we can use dovis
        # outside of a running simulation.
        gamma = self.cc_data.get_aux("gamma")

        myg = self.cc_data.grid

        q = flx.cons_to_prim_wrapper(self.cc_data.data, gamma, ivars, myg)

        rho = q[:, :, ivars.irho]
        u = q[:, :, ivars.iu]
        v = q[:, :, ivars.iv]
        p = q[:, :, ivars.ip]
        try:
            e = eos.rhoe(gamma, p)/rho
        except FloatingPointError:
            p[:, :] = self.cc_data.data[:, :, ivars.iener] * (gamma-1)
            e = self.cc_data.data[:, :, ivars.iener]  # p / (gamma - 1)

        magvel = np.sqrt(u**2 + v**2)

        fields = [rho, magvel, p, e]
        field_names = [r"$\rho$", r"$|U|$", "$p$", "$e$"]

        _, axes, cbar_title = plot_tools.setup_axes(myg, len(fields))

        for n, ax in enumerate(axes):
            v = fields[n]

            img = ax.imshow(np.transpose(v.v()),
                            interpolation="nearest", origin="lower",
                            extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
                            cmap=self.cm)

            ax.set_xlabel("x")
            ax.set_ylabel("y")

            # needed for PDF rendering
            cb = axes.cbar_axes[n].colorbar(img)
            cb.solids.set_rasterized(True)
            cb.solids.set_edgecolor("face")

            if cbar_title:
                cb.ax.set_title(field_names[n])
            else:
                ax.set_title(field_names[n])

        plt.figtext(0.05, 0.0125, f"t = {self.cc_data.t:10.5g}")

        plt.pause(0.001)
        plt.draw()

    def write_extras(self, f):
        """
        Output simulation-specific data to the h5py file f
        """

        # make note of the custom BC
        gb = f.create_group("BC")

        # the value here is the value of "is_solid"
        gb.create_dataset("hse", data=False)
