from __future__ import print_function

import importlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

import compressible.BC as BC
import compressible.eos as eos
import compressible.derives as derives
import mesh.boundary as bnd
import mesh.patch as patch
from simulation_null import NullSimulation, grid_setup, bc_setup
import compressible.unsplit_fluxes as flx

class Variables(object):
    """
    a container class for easy access to the different compressible
    variable by an integer key
    """
    def __init__(self, myd):
        self.nvar = len(myd.names)

        # conserved variables -- we set these when we initialize for
        # they match the CellCenterData2d object
        self.idens = myd.names.index("density")
        self.ixmom = myd.names.index("x-momentum")
        self.iymom = myd.names.index("y-momentum")
        self.iener = myd.names.index("energy")

        # if there are any additional variable, we treat them as
        # passively advected scalars
        self.naux = self.nvar - 4
        if self.naux > 0:
            self.irhox = 4
        else:
            self.irhox = -1

        # primitive variables
        self.irho = 0
        self.iu = 1
        self.iv = 2
        self.ip = 3

        if self.naux > 0:
            self.ix = 4   # advected scalar
        else:
            self.ix = -1


class Simulation(NullSimulation):

    def initialize(self, extra_vars=None):
        """
        Initialize the grid and variables for compressible flow and set
        the initial conditions for the chosen problem.
        """

        my_grid = grid_setup(self.rp, ng=4)
        my_data = patch.CellCenterData2d(my_grid)


        # define solver specific boundary condition routines
        bnd.define_bc("hse", BC.user, is_solid=False)

        bc, bc_xodd, bc_yodd = bc_setup(self.rp)

        # are we dealing with solid boundaries? we'll use these for
        # the Riemann solver
        self.solid = bnd.BCProp(int(bnd.bc_solid[self.rp.get_param("mesh.xlboundary")]),
                                int(bnd.bc_solid[self.rp.get_param("mesh.xrboundary")]),
                                int(bnd.bc_solid[self.rp.get_param("mesh.ylboundary")]),
                                int(bnd.bc_solid[self.rp.get_param("mesh.yrboundary")]))

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
        aux_data = patch.CellCenterData2d(my_grid)
        aux_data.register_var("ymom_src", bc_yodd)
        aux_data.register_var("E_src", bc)
        aux_data.create()
        self.aux_data = aux_data

        self.ivars = Variables(my_data)

        # derived variables
        self.cc_data.add_derived(derives.derive_primitives)

        # initial conditions for the problem
        problem = importlib.import_module("compressible.problems.{}".format(self.problem_name))
        problem.init_data(self.cc_data, self.rp)

        if self.verbose > 0: print(my_data)


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

        self.dt = cfl*min(xtmp.min(), ytmp.min())


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

            var.v()[:,:] += \
                dtdx*(Flux_x.v(n=n) - Flux_x.ip(1, n=n)) + \
                dtdy*(Flux_y.v(n=n) - Flux_y.jp(1, n=n))

        # gravitational source terms
        ymom[:,:] += 0.5*self.dt*(dens[:,:] + old_dens[:,:])*grav
        ener[:,:] += 0.5*self.dt*(ymom[:,:] + old_ymom[:,:])*grav

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

        dens = self.cc_data.get_var("density")

        nvar = len(self.cc_data.names)

        # get the velocities
        u, v = self.cc_data.get_var("velocity")
        magvel = np.sqrt(u**2 + v**2)

        # thermodynamic information
        e = self.cc_data.get_var("eint")

        # access gamma from the cc_data object so we can use dovis
        # outside of a running simulation.
        gamma = self.cc_data.get_aux("gamma")

        p = eos.pres(gamma, dens, e)

        myg = self.cc_data.grid

        # figure out the geometry
        L_x = self.cc_data.grid.xmax - self.cc_data.grid.xmin
        L_y = self.cc_data.grid.ymax - self.cc_data.grid.ymin

        f = plt.figure(1)

        cbar_title = False

        if L_x > 2*L_y:
            # we want 4 rows:
            axes = AxesGrid(f, 111,
                            nrows_ncols=(4, 1),
                            share_all=True,
                            cbar_mode="each",
                            cbar_location="top",
                            cbar_pad="10%",
                            cbar_size="25%",
                            axes_pad=(0.25, 0.65),
                            add_all=True, label_mode="L")
            cbar_title = True

        elif L_y > 2*L_x:
            # we want 4 columns:  rho  |U|  p  e
            axes = AxesGrid(f, 111,
                            nrows_ncols=(1, 4),
                            share_all=True,
                            cbar_mode="each",
                            cbar_location="right",
                            cbar_pad="10%",
                            cbar_size="25%",
                            axes_pad=(0.65, 0.25),
                            add_all=True, label_mode="L")

        else:
            # 2x2 grid of plots
            axes = AxesGrid(f, 111,
                            nrows_ncols=(2, 2),
                            share_all=True,
                            cbar_mode="each",
                            cbar_location="right",
                            cbar_pad="2%",
                            axes_pad=(0.65, 0.25),
                            add_all=True, label_mode="L")

        fields = [dens, magvel, p, e]
        field_names = [r"$\rho$", r"U", "p", "e"]

        cm = "viridis"

        for n in range(4):
            ax = axes[n]

            v = fields[n]
            img = ax.imshow(np.transpose(v.v()),
                            interpolation="nearest", origin="lower",
                            extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
                            cmap=cm)

            ax.set_xlabel("x")
            ax.set_ylabel("y")

            cb = axes.cbar_axes[n].colorbar(img)
            if cbar_title:
                cb.ax.set_title(field_names[n])
            else:
                ax.set_title(field_names[n])

        plt.figtext(0.05, 0.0125, "t = %10.5f" % self.cc_data.t)

        plt.pause(0.001)
        plt.draw()
