import numpy
import pylab

from diffusion.problems import *
import mesh.patch as patch
import multigrid.multigrid as multigrid
from util import msg, profile, runparams

class Simulation:

    def __init__(self, problem_name, rp, timers=None):
        """
        Initialize the Simulation object for diffusion:

           a  = k a
            t      xx

        Parameters
        ----------
        problem_name : str
            The name of the problem we wish to run.  This should
            correspond to one of the modules in diffusion/problems/
        rp : RuntimeParameters object
            The runtime parameters for the simulation
        timers : TimerCollection object, optional
            The timers used for profiling this simulation
        """

        self.rp = rp
        self.cc_data = None

        self.problem_name = problem_name

        if timers == None:
            self.tc = profile.TimerCollection()
        else:
            self.tc = timers


    def initialize(self):
        """ 
        Initialize the grid and variables for diffusion and set the initial
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
                               ymin=ymin, ymax=ymax, ng=1)


        # create the variables

        # first figure out the boundary conditions -- we allow periodic,
        # Dirichlet, and Neumann.

        xlb_type = self.rp.get_param("mesh.xlboundary")
        xrb_type = self.rp.get_param("mesh.xrboundary")
        ylb_type = self.rp.get_param("mesh.ylboundary")
        yrb_type = self.rp.get_param("mesh.yrboundary")

        bcparam = []
        for bc in [xlb_type, xrb_type, ylb_type, yrb_type]:
            if bc == "periodic": bcparam.append("periodic")
            elif bc == "neumann":  bcparam.append("neumann")
            elif bc == "dirichlet":  bcparam.append("dirichlet")
            else:
                msg.fail("invalid BC")


        bc = patch.BCObject(xlb=bcparam[0], xrb=bcparam[1], 
                            ylb=bcparam[2], yrb=bcparam[3])    


        my_data = patch.CellCenterData2d(my_grid)

        my_data.register_var("phi", bc)

        my_data.create()

        self.cc_data = my_data

        # now set the initial conditions for the problem           
        exec self.problem_name + '.init_data(self.cc_data, self.rp)'


    def timestep(self):
        """
        The diffusion timestep() function computes the timestep 
        using the explicit timestep constraint as the starting point.  
        We then multiply by the CFL number to get the timestep.  
        Since we are doing an implicit discretization, we do not 
        require CFL < 1.
        """
        
        cfl = self.rp.get_param("driver.cfl")
        k = self.rp.get_param("diffusion.k")
    
        # the timestep is min(dx**2/k, dy**2/k)
        xtmp = self.cc_data.grid.dx**2/k
        ytmp = self.cc_data.grid.dy**2/k

        dt = cfl*min(xtmp, ytmp)

        return dt


    def preevolve(myd):
        """
        Do any necessary evolution before the main evolve loop.  This
        is not needed for diffusion.
        """    
        pass


    def evolve(self, dt):
        """ 
        Diffusion through dt using C-N implicit solve with multigrid 
        """

        self.cc_data.fill_BC_all()
        phi = self.cc_data.get_var("phi")
        myg = self.cc_data.grid

        # diffusion coefficient
        k = self.rp.get_param("diffusion.k")
    

        # setup the MG object -- we want to solve a Helmholtz equation
        # equation of the form:
        # (alpha - beta L) phi = f
        #
        # with alpha = 1
        #      beta  = (dt/2) k
        #      f     = phi + (dt/2) k L phi
        #
        # this is the form that arises with a Crank-Nicolson discretization
        # of the diffusion equation.
        mg = multigrid.CellCenterMG2d(myg.nx, myg.ny,
                                      xmin=myg.xmin, xmax=myg.xmax, 
                                      ymin=myg.ymin, ymax=myg.ymax,
                                      xl_BC_type=self.cc_data.BCs['phi'].xlb, 
                                      xr_BC_type=self.cc_data.BCs['phi'].xrb, 
                                      yl_BC_type=self.cc_data.BCs['phi'].ylb, 
                                      yr_BC_type=self.cc_data.BCs['phi'].yrb, 
                                      alpha=1.0, beta=0.5*dt*k, 
                                      verbose=0)

        # form the RHS: f = phi + (dt/2) k L phi  (where L is the Laplacian)
        f = mg.soln_grid.scratch_array()
        f[mg.ilo:mg.ihi+1,mg.jlo:mg.jhi+1] = \
           phi[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] + 0.5*dt*k * \
           ((phi[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] +
             phi[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1] -
             2.0*phi[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx**2 +
            (phi[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] +
             phi[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ] -
             2.0*phi[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dy**2)

        mg.init_RHS(f)

        # initial guess is zeros
        mg.init_zeros()

        # solve the MG problem for the updated phi
        mg.solve(rtol=1.e-10)
        #mg.smooth(mg.nlevels-1,100)

        # update the solution
        phi[:,:] = mg.get_solution()

        
    def dovis(self):
        """
        Do runtime visualization. 
        """

        pylab.clf()

        phi = self.cc_data.get_var("phi")

        myg = self.cc_data.grid

        pylab.imshow(numpy.transpose(phi[myg.ilo:myg.ihi+1,
                                         myg.jlo:myg.jhi+1]), 
                     interpolation="nearest", origin="lower",
                     extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

        pylab.xlabel("x")
        pylab.ylabel("y")
        pylab.title("phi")

        pylab.colorbar()
        
        pylab.figtext(0.05,0.0125, "t = %10.5f" % self.cc_data.t)

        pylab.draw()


    def finalize(self):
        """
        Do any final clean-ups for the simulation and call the problem's
        finalize() method.
        """
        exec self.problem_name + '.finalize()'


