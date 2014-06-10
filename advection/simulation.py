import numpy
import pylab

from advection.problems import *
from advectiveFluxes import *
import mesh.patch as patch

class Simulation:

    def __init__(self, problem_name, rp):

        self.rp = rp
        self.cc_data = None

        self.SMALL = 1.e-12

        self.problem_name = problem_name


    def initialize(self):
        """ 
        Initialize the grid and variables for advection and set the initial
        conditions for the chosen problem.
        """

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

        # first figure out the boundary conditions -- we need to translate
        # between the descriptive type of the boundary specified by the
        # user and the action that will be performed by the fill_BC routine.
        # Usually the actions can vary depending on the variable, but we
        # only have one variable.
        xlb_type = self.rp.get_param("mesh.xlboundary")
        xrb_type = self.rp.get_param("mesh.xrboundary")
        ylb_type = self.rp.get_param("mesh.ylboundary")
        yrb_type = self.rp.get_param("mesh.yrboundary")
        
        bc = patch.BCObject(xlb=xlb_type, xrb=xrb_type, 
                            ylb=ylb_type, yrb=yrb_type)

        my_data = patch.CellCenterData2d(my_grid)

        my_data.register_var("density", bc)

        my_data.create()

        self.cc_data = my_data

        # now set the initial conditions for the problem
        exec self.problem_name + '.init_data(self.cc_data, self.rp)'


    def timestep(self):
        """
        Computes the advective timestep (CFL) constraint.  We use the 
        driver.cfl parameter to control what fraction of the CFL
        step we actually take.
        """
    
        cfl = self.rp.get_param("driver.cfl")
    
        u = self.rp.get_param("advection.u")
        v = self.rp.get_param("advection.v")
    
        # the timestep is min(dx/|u|, dy|v|)
        xtmp = self.cc_data.grid.dx/max(abs(u),self.SMALL)
        ytmp = self.cc_data.grid.dy/max(abs(v),self.SMALL)

        dt = cfl*min(xtmp, ytmp)

        return dt


    def preevolve(self):
    
        # do nothing
        pass


    def evolve(self, dt):
        """ 
        Evolve the linear advection equation through one timestep.  We only
        consider the "density" variable in the CellCenterData2d object input
        here.

        Parameters
        ----------
        self.cc_data : CellCenterData2d object
            The data object containing the scalar quantity we are advecting
        dt : float
            The timestep to evolve through
        
        """

        dtdx = dt/self.cc_data.grid.dx
        dtdy = dt/self.cc_data.grid.dy

        flux_x, flux_y =  unsplitFluxes(self.cc_data, self.rp, dt, "density")

        """
        do the differencing for the fluxes now.  Here, we use slices so we
        avoid slow loops in python.  This is equivalent to:

        myPatch.data[i,j] = myPatch.data[i,j] + \
                               dtdx*(flux_x[i,j] - flux_x[i+1,j]) + \
                               dtdy*(flux_y[i,j] - flux_y[i,j+1])
        """

        qx = self.cc_data.grid.qx
        qy = self.cc_data.grid.qy

        dens = self.cc_data.get_var("density")

        dens[0:qx-1,0:qy-1] = dens[0:qx-1,0:qy-1] + \
                   dtdx*(flux_x[0:qx-1,0:qy-1] - flux_x[1:qx,0:qy-1]) + \
                   dtdy*(flux_y[0:qx-1,0:qy-1] - flux_y[0:qx-1,1:qy])  


    def dovis(self, n):

        pylab.clf()

        dens = self.cc_data.get_var("density")

        myg = self.cc_data.grid

        pylab.imshow(numpy.transpose(dens[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
                     interpolation="nearest", origin="lower",
                     extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

        pylab.xlabel("x")
        pylab.ylabel("y")
        pylab.title("density")

        pylab.colorbar()

        pylab.figtext(0.05,0.0125, "t = %10.5f" % self.cc_data.t)

        pylab.draw()

    
    def finalize(self):

        exec self.problem_name + '.finalize()'        
