import numpy as np
import matplotlib.pyplot as plt

from diffusion.problems import *
import mesh.patch as patch
from simulation_null import NullSimulation, grid_setup
import multigrid.MG as MG
from util import msg, profile

class Simulation(NullSimulation):

    def initialize(self):
        """
        Initialize the grid and variables for diffusion and set the initial
        conditions for the chosen problem.
        """

        # setup the grid
        my_grid = grid_setup(self.rp, ng=1)

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
        exec(self.problem_name + '.init_data(self.cc_data, self.rp)')


    def compute_timestep(self):
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

        self.dt = cfl*min(xtmp, ytmp)


    def evolve(self):
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
        mg = MG.CellCenterMG2d(myg.nx, myg.ny,
                               xmin=myg.xmin, xmax=myg.xmax,
                               ymin=myg.ymin, ymax=myg.ymax,
                               xl_BC_type=self.cc_data.BCs['phi'].xlb,
                               xr_BC_type=self.cc_data.BCs['phi'].xrb,
                               yl_BC_type=self.cc_data.BCs['phi'].ylb,
                               yr_BC_type=self.cc_data.BCs['phi'].yrb,
                               alpha=1.0, beta=0.5*self.dt*k,
                               verbose=0)

        # form the RHS: f = phi + (dt/2) k L phi  (where L is the Laplacian)
        f = mg.soln_grid.scratch_array()
        f.v()[:,:] = phi.v() + 0.5*self.dt*k * (
            (phi.ip(1) + phi.ip(-1) - 2.0*phi.v())/myg.dx**2 +
            (phi.jp(1) + phi.jp(-1) - 2.0*phi.v())/myg.dy**2)

        mg.init_RHS(f.d)

        # initial guess is zeros
        mg.init_zeros()

        # solve the MG problem for the updated phi
        mg.solve(rtol=1.e-10)
        #mg.smooth(mg.nlevels-1,100)

        # update the solution
        phi.v()[:,:] = mg.get_solution().v()

        # increment the time
        self.cc_data.t += self.dt
        self.n += 1


    def dovis(self):
        """
        Do runtime visualization.
        """

        plt.clf()

        phi = self.cc_data.get_var("phi")

        myg = self.cc_data.grid

        plt.imshow(np.transpose(phi[myg.ilo:myg.ihi+1,
                                         myg.jlo:myg.jhi+1]),
                     interpolation="nearest", origin="lower",
                     extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("phi")

        plt.colorbar()

        plt.figtext(0.05,0.0125, "t = %10.5f" % self.cc_data.t)

        plt.draw()
