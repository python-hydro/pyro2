from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import radhydro.eos as eos
import mesh.boundary as bnd
import mesh.patch as patch
import compressible.unsplitFluxes as flx
from util import profile


class Variables(object):
    """
    a container class for easy access to the different compressible
    variable by an integer key
    """
    def __init__(self, idens=-1, ixmom=-1, iymom=-1, iener=-1, ierad=-1):
        self.nvar = 5

        # conserved variables -- we set these when we initialize for
        # they match the CellCenterData2d object
        self.idens = idens
        self.ixmom = ixmom
        self.iymom = iymom
        self.iener = iener
        self.ierad = ierad
        
        # primitive variables
        self.irho = 0
        self.iu = 1
        self.iv = 2
        self.iptot = 3
        self.iretot = 4
        self.ire_g = 5

class Simulation:

    def __init__(self, problem_name, rp, timers=None):
        """
        Initialize the Simulation object for compressible hydrodynamics.

        Parameters
        ----------
        problem_name : str
            The name of the problem we wish to run.  This should
            correspond to one of the modules in compressible/problems/
        rp : RuntimeParameters object
            The runtime parameters for the simulation
        timers : TimerCollection object, optional
            The timers used for profiling this simulation
        """

        self.rp = rp
        self.cc_data = None
        self.problem_name = problem_name

        self.ivars = None

        self.SMALL = 1.e-12

        if timers == None:
            self.tc = profile.TimerCollection()
        else:
            self.tc = timers


    def initialize(self):
        """
        Initialize the grid and variables for compressible flow and set
        the initial conditions for the chosen problem.
        """

        # setup the grid
        nx = self.rp.get_param("mesh.nx")
        ny = self.rp.get_param("mesh.ny")

        xmin = self.rp.get_param("mesh.xmin")
        xmax = self.rp.get_param("mesh.xmax")
        ymin = self.rp.get_param("mesh.ymin")
        ymax = self.rp.get_param("mesh.ymax")

        verbose = self.rp.get_param("driver.verbose")
        
        my_grid = patch.Grid2d(nx, ny,
                               xmin=xmin, xmax=xmax,
                               ymin=ymin, ymax=ymax, ng=4)


        # create the variables
        my_data = patch.CellCenterData2d(my_grid)


        # define solver specific boundary condition routines
        patch.define_bc("hse", BC.user)


        # first figure out the boundary conditions.  Note: the action
        # can depend on the variable (for reflecting BCs)
        xlb_type = self.rp.get_param("mesh.xlboundary")
        xrb_type = self.rp.get_param("mesh.xrboundary")
        ylb_type = self.rp.get_param("mesh.ylboundary")
        yrb_type = self.rp.get_param("mesh.yrboundary")

        bc = bnd.BC(xlb=xlb_type, xrb=xrb_type,
                    ylb=ylb_type, yrb=yrb_type)

        # density and energy
        my_data.register_var("density", bc)
        my_data.register_var("energy", bc)
        my_data.register_var("erad", bc)        

        # for velocity, if we are reflecting, we need odd reflection
        # in the normal direction.

        # x-momentum -- if we are reflecting in x, then we need to
        # reflect odd
        bc_xodd = bnd.BC(xlb=xlb_type, xrb=xrb_type,
                         ylb=ylb_type, yrb=yrb_type,
                         odd_reflect_dir="x")

        my_data.register_var("x-momentum", bc_xodd)


        # y-momentum -- if we are reflecting in y, then we need to
        # reflect odd
        bc_yodd = bnd.BC(xlb=xlb_type, xrb=xrb_type,
                         ylb=ylb_type, yrb=yrb_type,
                         odd_reflect_dir="y")

        my_data.register_var("y-momentum", bc_yodd)


        # store grav because we'll need that in some BCs
        my_data.set_aux("grav", self.rp.get_param("compressible.grav"))

        my_data.create()

        self.cc_data = my_data

        self.ivars = Variables(idens = my_data.names.index("density"),
                               ixmom = my_data.names.index("x-momentum"),
                               iymom = my_data.names.index("y-momentum"),
                               iener = my_data.names.index("energy"),
                               ierad = my_data.names.index("erad"))        


        # initial conditions for the problem
        problem = importlib.import_module("radhydro.problems.{}".format(self.problem_name))
        problem.init_data(self.cc_data, self.rp)

        if verbose > 0: print(my_data)


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

        # HACK -- need to update -- this is not general
        # compute the sounds speed
        cs = np.sqrt(gamma*p/dens)


        # the timestep is min(dx/(|u| + cs), dy/(|v| + cs))
        xtmp = self.cc_data.grid.dx/(abs(u) + cs)
        ytmp = self.cc_data.grid.dy/(abs(v) + cs)

        dt = cfl*min(xtmp.min(), ytmp.min())

        return dt


    def preevolve(self):
        """
        Do any necessary evolution before the main evolve loop.  This
        is not needed for compressible flow.
        """
        pass


    def evolve(self, dt):
        """
        Evolve the equations of compressible hydrodynamics through a
        timestep dt.
        """

        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()

        dens = self.cc_data.get_var("density")
        xmom = self.cc_data.get_var("x-momentum")
        ymom = self.cc_data.get_var("y-momentum")
        ener = self.cc_data.get_var("energy")
        erad = self.cc_data.get_var("erad")        

        grav = self.rp.get_param("compressible.grav")

        myg = self.cc_data.grid

        Flux_x, Flux_y = flx.unsplitFluxes(self.cc_data, self.rp, self.ivars, self.tc, dt)

        old_dens = dens.copy()
        old_ymom = ymom.copy()

        # conservative update
        dtdx = dt/myg.dx
        dtdy = dt/myg.dy

        for n in range(self.ivars.nvar):
            var = self.cc_data.get_var_by_index(n)

            var[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] += \
                dtdx*(Flux_x[myg.ilo  :myg.ihi+1,myg.jlo  :myg.jhi+1,n] - \
                      Flux_x[myg.ilo+1:myg.ihi+2,myg.jlo  :myg.jhi+1,n]) + \
                dtdy*(Flux_y[myg.ilo  :myg.ihi+1,myg.jlo  :myg.jhi+1,n] - \
                      Flux_y[myg.ilo  :myg.ihi+1,myg.jlo+1:myg.jhi+2,n])

        # radiation source terms
        # HACK -- need to update
        
        # gravitational source terms
        ymom += 0.5*dt*(dens + old_dens)*grav
        ener += 0.5*dt*(ymom + old_ymom)*grav


        # parabolic update
        # HACK -- need to update
        
        
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

        # HACK -- this needs to be generalized
        
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

        if (L_x > 2*L_y):

            # we want 4 rows:
            #  rho
            #  |U|
            #   p
            #   e
            fig, axes = plt.subplots(nrows=4, ncols=1, num=1)
            orientation = "horizontal"
            if (L_x > 4*L_y):
                shrink = 0.75

            onLeft = list(range(self.ivars.nvar))


        elif (L_y > 2*L_x):

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
            img = ax.imshow(np.transpose(v[myg.ilo:myg.ihi+1,
                                           myg.jlo:myg.jhi+1]),
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


    def finalize(self):
        """
        Do any final clean-ups for the simulation and call the problem's
        finalize() method.
        """
        exec(self.problem_name + '.finalize()')
