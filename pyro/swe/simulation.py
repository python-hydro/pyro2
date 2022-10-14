import importlib

import matplotlib
import numpy as np

try:
    matplotlib.rcParams['mpl_toolkits.legacy_colorbar'] = False
except KeyError:
    pass
import matplotlib.pyplot as plt

import pyro.mesh.boundary as bnd
import pyro.particles.particles as particles
import pyro.swe.derives as derives
import pyro.swe.unsplit_fluxes as flx
import pyro.util.plot_tools as plot_tools
from pyro.simulation_null import NullSimulation, bc_setup, grid_setup


class Variables:
    """
    a container class for easy access to the different swe
    variables by an integer key
    """
    def __init__(self, myd):
        self.nvar = len(myd.names)

        # conserved variables -- we set these when we initialize for
        # they match the CellCenterData2d object
        self.ih = myd.names.index("height")
        self.ixmom = myd.names.index("x-momentum")
        self.iymom = myd.names.index("y-momentum")

        # if there are any additional variables, we treat them as
        # passively advected scalars
        self.naux = self.nvar - 3
        if self.naux > 0:
            self.ihx = 3
        else:
            self.ihx = -1

        # primitive variables
        self.nq = 3 + self.naux

        self.ih = 0
        self.iu = 1
        self.iv = 2

        if self.naux > 0:
            self.ix = 3   # advected scalar
        else:
            self.ix = -1


def cons_to_prim(U, g, ivars, myg):
    """
    Convert an input vector of conserved variables
    :math:`U = (h, hu, hv, {hX})`
    to primitive variables :math:`q = (h, u, v, {X})`.
    """

    q = myg.scratch_array(nvar=ivars.nq)

    q[:, :, ivars.ih] = U[:, :, ivars.ih]
    q[:, :, ivars.iu] = U[:, :, ivars.ixmom]/U[:, :, ivars.ih]
    q[:, :, ivars.iv] = U[:, :, ivars.iymom]/U[:, :, ivars.ih]

    if ivars.naux > 0:
        for nq, nu in zip(range(ivars.ix, ivars.ix+ivars.naux),
                          range(ivars.ihx, ivars.ihx+ivars.naux)):
            q[:, :, nq] = U[:, :, nu]/q[:, :, ivars.ih]

    return q


def prim_to_cons(q, g, ivars, myg):
    """
    Convert an input vector of primitive variables :math:`q = (h, u, v, {X})`
    to conserved variables :math:`U = (h, hu, hv, {hX})`
    """

    U = myg.scratch_array(nvar=ivars.nvar)

    U[:, :, ivars.ih] = q[:, :, ivars.ih]
    U[:, :, ivars.ixmom] = q[:, :, ivars.iu]*U[:, :, ivars.ih]
    U[:, :, ivars.iymom] = q[:, :, ivars.iv]*U[:, :, ivars.ih]

    if ivars.naux > 0:
        for nq, nu in zip(range(ivars.ix, ivars.ix+ivars.naux),
                          range(ivars.ihx, ivars.ihx+ivars.naux)):
            U[:, :, nu] = q[:, :, nq]*q[:, :, ivars.ih]

    return U


class Simulation(NullSimulation):
    """The main simulation class for the corner transport upwind
    swe hydrodynamics solver

    """

    def initialize(self, extra_vars=None, ng=4):
        """
        Initialize the grid and variables for swe flow and set
        the initial conditions for the chosen problem.
        """
        my_grid = grid_setup(self.rp, ng=ng)
        my_data = self.data_class(my_grid)

        bc, bc_xodd, bc_yodd = bc_setup(self.rp)

        # are we dealing with solid boundaries? we'll use these for
        # the Riemann solver
        self.solid = bnd.bc_is_solid(bc)

        # density and energy
        my_data.register_var("height", bc)
        my_data.register_var("x-momentum", bc_xodd)
        my_data.register_var("y-momentum", bc_yodd)
        my_data.register_var("fuel", bc)

        # any extras?
        if extra_vars is not None:
            for v in extra_vars:
                my_data.register_var(v, bc)

        # store the gravitational acceration g as an auxillary quantity
        # so we can have a
        # self-contained object stored in output files to make plots.
        # store grav because we'll need that in some BCs
        my_data.set_aux("g", self.rp.get_param("swe.grav"))

        my_data.create()

        self.cc_data = my_data

        if self.rp.get_param("particles.do_particles") == 1:
            n_particles = self.rp.get_param("particles.n_particles")
            particle_generator = self.rp.get_param("particles.particle_generator")
            self.particles = particles.Particles(self.cc_data, bc, n_particles, particle_generator)

        # some auxillary data that we'll need to fill GC in, but isn't
        # really part of the main solution
        aux_data = self.data_class(my_grid)
        aux_data.register_var("ymom_src", bc_yodd)
        aux_data.create()
        self.aux_data = aux_data

        self.ivars = Variables(my_data)

        # derived variables
        self.cc_data.add_derived(derives.derive_primitives)

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

        # the timestep is min(dx/(|u| + cs), dy/(|v| + cs))
        xtmp = self.cc_data.grid.dx/(abs(u) + cs)
        ytmp = self.cc_data.grid.dy/(abs(v) + cs)

        self.dt = cfl*float(min(xtmp.min(), ytmp.min()))

    def evolve(self):
        """
        Evolve the equations of swe hydrodynamics through a
        timestep dt.
        """

        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()

        myg = self.cc_data.grid

        Flux_x, Flux_y = flx.unsplit_fluxes(self.cc_data, self.aux_data, self.rp,
                                            self.ivars, self.solid, self.tc, self.dt)

        # conservative update
        dtdx = self.dt/myg.dx
        dtdy = self.dt/myg.dy

        for n in range(self.ivars.nvar):
            var = self.cc_data.get_var_by_index(n)

            var.v()[:, :] += \
                dtdx*(Flux_x.v(n=n) - Flux_x.ip(1, n=n)) + \
                dtdy*(Flux_y.v(n=n) - Flux_y.jp(1, n=n))

        if self.particles is not None:
            self.particles.update_particles(self.dt)

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

        # access g from the cc_data object so we can use dovis
        # outside of a running simulation.
        g = self.cc_data.get_aux("g")

        q = cons_to_prim(self.cc_data.data, g, ivars, self.cc_data.grid)

        h = q[:, :, ivars.ih]
        u = q[:, :, ivars.iu]
        v = q[:, :, ivars.iv]
        fuel = q[:, :, ivars.ix]

        magvel = np.sqrt(u**2 + v**2)

        myg = self.cc_data.grid

        vort = myg.scratch_array()

        dv = 0.5*(v.ip(1) - v.ip(-1))/myg.dx
        du = 0.5*(u.jp(1) - u.jp(-1))/myg.dy

        vort.v()[:, :] = dv - du

        fields = [h, magvel, fuel, vort]
        field_names = [r"$h$", r"$|U|$", r"$X$", r"$\nabla\times U$"]

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
            cb.formatter = matplotlib.ticker.FormatStrFormatter("")
            cb.solids.set_rasterized(True)
            cb.solids.set_edgecolor("face")

            if cbar_title:
                cb.ax.set_title(field_names[n])
            else:
                ax.set_title(field_names[n])

        if self.particles is not None:
            ax = axes[0]
            particle_positions = self.particles.get_positions()
            # dye particles
            colors = self.particles.get_init_positions()[:, 0]

            # plot particles
            ax.scatter(particle_positions[:, 0],
                particle_positions[:, 1], s=5, c=colors, alpha=0.8, cmap="Greys")
            ax.set_xlim([myg.xmin, myg.xmax])
            ax.set_ylim([myg.ymin, myg.ymax])

        plt.figtext(0.05, 0.0125, f"t = {self.cc_data.t:10.5g}")

        plt.pause(0.001)
        plt.draw()
