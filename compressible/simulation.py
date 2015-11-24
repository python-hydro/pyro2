from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import compressible.BC as BC
from compressible.problems import *
import compressible.eos as eos
import mesh.patch as patch
from simulation_null import NullSimulation, grid_setup, bc_setup
from compressible.unsplitFluxes import *
from util import profile


class Variables(object):
    """
    a container class for easy access to the different compressible
    variable by an integer key
    """
    def __init__(self, idens=-1, ixmom=-1, iymom=-1, iener=-1):
        self.nvar = 4

        # conserved variables -- we set these when we initialize for
        # they match the CellCenterData2d object
        self.idens = idens
        self.ixmom = ixmom
        self.iymom = iymom
        self.iener = iener

        # primitive variables
        self.irho = 0
        self.iu = 1
        self.iv = 2
        self.ip = 3


class BCProp(object):
    """
    a simple container to hold properties of the boundary conditions
    """
    def __init__(self, xl_prop, xr_prop, yl_prop, yr_prop):
        self.xl = xl_prop
        self.xr = xr_prop
        self.yl = yl_prop
        self.yr = yr_prop


class Simulation(NullSimulation):

    def initialize(self):
        """
        Initialize the grid and variables for compressible flow and set
        the initial conditions for the chosen problem.
        """
        
        my_grid = grid_setup(self.rp, ng=4)
        my_data = patch.CellCenterData2d(my_grid)


        # define solver specific boundary condition routines
        patch.define_bc("hse", BC.user, is_solid=False)

        bc, bc_xodd, bc_yodd = bc_setup(self.rp)

        # are we dealing with solid boundaries? we'll use these for
        # the Riemann solver
        self.solid = BCProp(int(patch.bc_props[self.rp.get_param("mesh.xlboundary")]),
                            int(patch.bc_props[self.rp.get_param("mesh.xrboundary")]),
                            int(patch.bc_props[self.rp.get_param("mesh.ylboundary")]),
                            int(patch.bc_props[self.rp.get_param("mesh.yrboundary")]))

        # density and energy
        my_data.register_var("density", bc)
        my_data.register_var("energy", bc)
        my_data.register_var("x-momentum", bc_xodd)
        my_data.register_var("y-momentum", bc_yodd)


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

        self.vars = Variables(idens = my_data.vars.index("density"),
                              ixmom = my_data.vars.index("x-momentum"),
                              iymom = my_data.vars.index("y-momentum"),
                              iener = my_data.vars.index("energy"))


        # initial conditions for the problem
        exec(self.problem_name + '.init_data(self.cc_data, self.rp)')

        if self.verbose > 0: print(my_data)


    def compute_timestep(self):
        """
        The timestep function computes the advective timestep (CFL)
        constraint.  The CFL constraint says that information cannot
        propagate further than one zone per timestep.

        We use the driver.cfl parameter to control what fraction of the
        CFL step we actually take.
        """

        cfl = self.rp.get_param("driver.cfl")

        # get the variables we need
        dens = self.cc_data.get_var("density")
        xmom = self.cc_data.get_var("x-momentum")
        ymom = self.cc_data.get_var("y-momentum")
        ener = self.cc_data.get_var("energy")

        # we need to compute the pressure
        u = xmom/dens
        v = ymom/dens

        e = (ener - 0.5*dens*(u*u + v*v))/dens

        gamma = self.rp.get_param("eos.gamma")

        p = eos.pres(gamma, dens, e)

        # compute the sounds speed
        cs = np.sqrt(gamma*p/dens)


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

        Flux_x, Flux_y = unsplitFluxes(self.cc_data, self.aux_data, self.rp, 
                                       self.vars, self.solid, self.tc, self.dt)

        old_dens = dens.copy()
        old_ymom = ymom.copy()

        # conservative update
        dtdx = self.dt/myg.dx
        dtdy = self.dt/myg.dy

        for n in range(self.vars.nvar):
            var = self.cc_data.get_var_by_index(n)

            var.v()[:,:] += \
                dtdx*(Flux_x.v(n=n) - Flux_x.ip(1, n=n)) + \
                dtdy*(Flux_y.v(n=n) - Flux_y.jp(1, n=n))

        # gravitational source terms
        ymom.d[:,:] += 0.5*self.dt*(dens.d[:,:] + old_dens.d[:,:])*grav
        ener.d[:,:] += 0.5*self.dt*(ymom.d[:,:] + old_ymom.d[:,:])*grav

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
        xmom = self.cc_data.get_var("x-momentum")
        ymom = self.cc_data.get_var("y-momentum")
        ener = self.cc_data.get_var("energy")

        # get the velocities
        u = xmom/dens
        v = ymom/dens

        # get the pressure
        magvel = u**2 + v**2   # temporarily |U|^2
        rhoe = (ener - 0.5*dens*magvel)

        magvel = np.sqrt(magvel)

        e = rhoe/dens

        # access gamma from the cc_data object so we can use dovis
        # outside of a running simulation.
        gamma = self.cc_data.get_aux("gamma")

        p = eos.pres(gamma, dens, e)

        myg = self.cc_data.grid


        # figure out the geometry
        L_x = self.cc_data.grid.xmax - self.cc_data.grid.xmin
        L_y = self.cc_data.grid.ymax - self.cc_data.grid.ymin

        orientation = "vertical"
        shrink = 1.0

        sparseX = 0
        allYlabel = 1

        if L_x > 2*L_y:

            # we want 4 rows:
            #  rho
            #  |U|
            #   p
            #   e
            fig, axes = plt.subplots(nrows=4, ncols=1, num=1)
            orientation = "horizontal"
            if (L_x > 4*L_y):
                shrink = 0.75

            onLeft = list(range(self.vars.nvar))


        elif L_y > 2*L_x:

            # we want 4 columns:  rho  |U|  p  e
            fig, axes = plt.subplots(nrows=1, ncols=4, num=1)
            if (L_y >= 3*L_x):
                shrink = 0.5
                sparseX = 1
                allYlabel = 0

            onLeft = [0]

        else:
            # 2x2 grid of plots with
            #
            #   rho   |u|
            #    p     e
            fig, axes = plt.subplots(nrows=2, ncols=2, num=1)
            plt.subplots_adjust(hspace=0.25)

            onLeft = [0,2]


        fields = [dens, magvel, p, e]
        field_names = [r"$\rho$", r"U", "p", "e"]

        for n in range(4):
            ax = axes.flat[n]

            v = fields[n]
            img = ax.imshow(np.transpose(v.v()),
                        interpolation="nearest", origin="lower",
                        extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

            ax.set_xlabel("x")
            if n == 0:
                ax.set_ylabel("y")
            elif allYlabel:
                ax.set_ylabel("y")

            ax.set_title(field_names[n])

            if not n in onLeft:
                ax.yaxis.offsetText.set_visible(False)
                if n > 0: ax.get_yaxis().set_visible(False)

            if sparseX:
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))

            plt.colorbar(img, ax=ax, orientation=orientation, shrink=shrink)


        plt.figtext(0.05,0.0125, "t = %10.5f" % self.cc_data.t)

        plt.draw()
