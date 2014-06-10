import numpy
import pylab
from matplotlib.font_manager import FontProperties

import BC
from compressible.problems import *
import eos
import mesh.patch as patch
from unsplitFluxes import *
from util import profile

class Simulation:

    def __init__(self, problem_name, rp):

        self.rp = rp
        self.cc_data = None
        self.problem_name = problem_name

        self.SMALL = 1.e-12

    def initialize(self):
        """ 
        initialize the grid and variables for compressible flow
        """
        import vars

        # setup the grid
        nx = self.rp.get_param("mesh.nx")
        ny = self.rp.get_param("mesh.ny")

        xmin = self.rp.get_param("mesh.xmin")
        xmax = self.rp.get_param("mesh.xmax")
        ymin = self.rp.get_param("mesh.ymin")
        ymax = self.rp.get_param("mesh.ymax")
    
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
    
        bc = patch.BCObject(xlb=xlb_type, xrb=xrb_type,
                            ylb=ylb_type, yrb=yrb_type)

        # density and energy
        my_data.register_var("density", bc)
        my_data.register_var("energy", bc)

        # for velocity, if we are reflecting, we need odd reflection
        # in the normal direction.

        # x-momentum -- if we are reflecting in x, then we need to
        # reflect odd
        bc_xodd = patch.BCObject(xlb=xlb_type, xrb=xrb_type,
                                 ylb=ylb_type, yrb=yrb_type,
                                 odd_reflect_dir="x")

        my_data.register_var("x-momentum", bc_xodd)    


        # y-momentum -- if we are reflecting in y, then we need to
        # reflect odd
        bc_yodd = patch.BCObject(xlb=xlb_type, xrb=xrb_type,
                                 ylb=ylb_type, yrb=yrb_type,
                                 odd_reflect_dir="y")

        my_data.register_var("y-momentum", bc_yodd)    


        # store the EOS gamma as an auxillary quantity so we can have a
        # self-contained object stored in output files to make plots
        gamma = self.rp.get_param("eos.gamma")
        my_data.set_aux("gamma", gamma)

        # initialize the EOS gamma
        eos.init(gamma)

        my_data.create()

        self.cc_data = my_data

        vars.idens = my_data.vars.index("density")
        vars.ixmom = my_data.vars.index("x-momentum")
        vars.iymom = my_data.vars.index("y-momentum")
        vars.iener = my_data.vars.index("energy")

        # initial conditions for the problem
        exec self.problem_name + '.init_data(self.cc_data, self.rp)'

        print my_data


    def timestep(self):
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

        p = eos.pres(dens, e)

        # compute the sounds speed
        gamma = self.rp.get_param("eos.gamma")

        cs = numpy.sqrt(gamma*p/dens)


        # the timestep is min(dx/(|u| + cs), dy/(|v| + cs))
        xtmp = self.cc_data.grid.dx/(abs(u) + cs)
        ytmp = self.cc_data.grid.dy/(abs(v) + cs)

        dt = cfl*min(xtmp.min(), ytmp.min())

        return dt


    def preevolve(self):
    
        # do nothing
        pass


    def evolve(self, dt):

        pf = profile.timer("evolve")
        pf.begin()

        dens = self.cc_data.get_var("density")
        xmom = self.cc_data.get_var("x-momentum")
        ymom = self.cc_data.get_var("y-momentum")
        ener = self.cc_data.get_var("energy")

        grav = self.rp.get_param("compressible.grav")

        myg = self.cc_data.grid
        
        Flux_x, Flux_y = unsplitFluxes(self.cc_data, self.rp, dt)

        old_dens = dens.copy()
        old_ymom = ymom.copy()

        # conservative update
        dtdx = dt/myg.dx
        dtdy = dt/myg.dy

        dens[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] += \
           dtdx*(Flux_x[myg.ilo  :myg.ihi+1,
                        myg.jlo  :myg.jhi+1,vars.idens] - \
                 Flux_x[myg.ilo+1:myg.ihi+2,
                        myg.jlo  :myg.jhi+1,vars.idens]) + \
           dtdy*(Flux_y[myg.ilo  :myg.ihi+1,
                        myg.jlo  :myg.jhi+1,vars.idens] - \
                 Flux_y[myg.ilo  :myg.ihi+1,
                        myg.jlo+1:myg.jhi+2,vars.idens])    

        xmom[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] += \
           dtdx*(Flux_x[myg.ilo  :myg.ihi+1,
                        myg.jlo  :myg.jhi+1,vars.ixmom] - \
                 Flux_x[myg.ilo+1:myg.ihi+2,
                        myg.jlo  :myg.jhi+1,vars.ixmom]) + \
           dtdy*(Flux_y[myg.ilo  :myg.ihi+1,
                        myg.jlo  :myg.jhi+1,vars.ixmom] - \
                 Flux_y[myg.ilo  :myg.ihi+1,
                        myg.jlo+1:myg.jhi+2,vars.ixmom])    

        ymom[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] += \
           dtdx*(Flux_x[myg.ilo  :myg.ihi+1,
                        myg.jlo  :myg.jhi+1,vars.iymom] - \
                 Flux_x[myg.ilo+1:myg.ihi+2,
                        myg.jlo  :myg.jhi+1,vars.iymom]) + \
           dtdy*(Flux_y[myg.ilo  :myg.ihi+1,
                        myg.jlo  :myg.jhi+1,vars.iymom] - \
                 Flux_y[myg.ilo  :myg.ihi+1,
                        myg.jlo+1:myg.jhi+2,vars.iymom])    

        ener[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] += \
           dtdx*(Flux_x[myg.ilo  :myg.ihi+1,
                        myg.jlo  :myg.jhi+1,vars.iener] - \
                 Flux_x[myg.ilo+1:myg.ihi+2,
                        myg.jlo  :myg.jhi+1,vars.iener]) + \
           dtdy*(Flux_y[myg.ilo  :myg.ihi+1,
                        myg.jlo  :myg.jhi+1,vars.iener] - \
                 Flux_y[myg.ilo  :myg.ihi+1,
                        myg.jlo+1:myg.jhi+2,vars.iener])    


        # gravitational source terms
        ymom += 0.5*dt*(dens + old_dens)*grav
        ener += 0.5*dt*(ymom + old_ymom)*grav

        pf.end()


    def dovis(self, n):

        pylab.clf()

        pylab.rc("font", size=10)

        dens = self.cc_data.get_var("density")
        xmom = self.cc_data.get_var("x-momentum")
        ymom = self.cc_data.get_var("y-momentum")
        ener = self.cc_data.get_var("energy")

        nvar = 4
    
        # get the velocities
        u = xmom/dens
        v = ymom/dens

        # get the pressure
        magvel = u**2 + v**2   # temporarily |U|^2
        rhoe = (ener - 0.5*dens*magvel)

        magvel = numpy.sqrt(magvel)

        # access gamma from the object instead of using the EOS so we can
        # use dovis outside of a running simulation.
        gamma = self.cc_data.get_aux("gamma")
        p = rhoe*(gamma - 1.0)

        e = rhoe/dens

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
            fig, axes = pylab.subplots(nrows=4, ncols=1, num=1)
            orientation = "horizontal"
            if (L_x > 4*L_y):
                shrink = 0.75

            onLeft = list(range(nvar))


        elif (L_y > 2*L_x):

            # we want 4 columns:
            # 
            #  rho  |U|  p  e
            fig, axes = pylab.subplots(nrows=1, ncols=4, num=1)        
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
            fig, axes = pylab.subplots(nrows=2, ncols=2, num=1)
            pylab.subplots_adjust(hspace=0.25)

            onLeft = [0,2]

        ax = axes.flat[0]

        img = ax.imshow(numpy.transpose(dens[myg.ilo:myg.ihi+1,
                                             myg.jlo:myg.jhi+1]), 
                        interpolation="nearest", origin="lower",
                        extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(r"$\rho$")

        if not 0 in onLeft:
            ax.yaxis.offsetText.set_visible(False)
        
        if sparseX:
            ax.xaxis.set_major_locator(pylab.MaxNLocator(3))

        pylab.colorbar(img, ax=ax, orientation=orientation, shrink=shrink)

        ax = axes.flat[1]

        img = ax.imshow(numpy.transpose(magvel[myg.ilo:myg.ihi+1,
                                               myg.jlo:myg.jhi+1]), 
                        interpolation="nearest", origin="lower",
                        extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

        ax.set_xlabel("x")
        if (allYlabel): ax.set_ylabel("y")
        ax.set_title("U")

        if not 1 in onLeft:
            ax.get_yaxis().set_visible(False)
            ax.yaxis.offsetText.set_visible(False)

        if sparseX:
            ax.xaxis.set_major_locator(pylab.MaxNLocator(3))

        pylab.colorbar(img, ax=ax, orientation=orientation, shrink=shrink)

        ax = axes.flat[2]

        img = ax.imshow(numpy.transpose(p[myg.ilo:myg.ihi+1,
                                          myg.jlo:myg.jhi+1]), 
                        interpolation="nearest", origin="lower",
                        extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

        ax.set_xlabel("x")
        if (allYlabel): ax.set_ylabel("y")
        ax.set_title("p")

        if not 2 in onLeft:
            ax.get_yaxis().set_visible(False)
            ax.yaxis.offsetText.set_visible(False)

        if sparseX:
            ax.xaxis.set_major_locator(pylab.MaxNLocator(3))

        pylab.colorbar(img, ax=ax, orientation=orientation, shrink=shrink)

        ax = axes.flat[3]

        img = ax.imshow(numpy.transpose(e[myg.ilo:myg.ihi+1,
                                          myg.jlo:myg.jhi+1]), 
                        interpolation="nearest", origin="lower",
                        extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

        ax.set_xlabel("x")
        if (allYlabel): ax.set_ylabel("y")
        ax.set_title("e")

        if not 3 in onLeft:
            ax.get_yaxis().set_visible(False)
            ax.yaxis.offsetText.set_visible(False)

        if sparseX:
            ax.xaxis.set_major_locator(pylab.MaxNLocator(3))

        pylab.colorbar(img, ax=ax, orientation=orientation, shrink=shrink)

        pylab.figtext(0.05,0.0125, "t = %10.5f" % self.cc_data.t)

        pylab.draw()


    def finalize(self):

        exec self.problem_name + '.finalize()'

