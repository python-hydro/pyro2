import numpy as np
import matplotlib.pyplot as plt

from advection.problems import *
from advection.advectiveFluxes import *
import mesh.patch as patch
from simulation_null import NullSimulation, grid_setup, bc_setup
from util import profile

class Simulation(NullSimulation):

    def initialize(self):
        """
        Initialize the grid and variables for advection and set the initial
        conditions for the chosen problem.
        """

        my_grid = grid_setup(self.rp, ng=4)

        # create the variables
        my_data = patch.CellCenterData2d(my_grid)
        bc = bc_setup(self.rp)[0]
        my_data.register_var("density", bc)
        my_data.create()

        self.cc_data = my_data

        # now set the initial conditions for the problem
        exec(self.problem_name + '.init_data(self.cc_data, self.rp)')


    def timestep(self):
        """
        Compute the advective timestep (CFL) constraint.  We use the
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


    def evolve(self, dt):
        """
        Evolve the linear advection equation through one timestep.  We only
        consider the "density" variable in the CellCenterData2d object that
        is part of the Simulation.

        Parameters
        ----------
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

        dens.d[0:qx-1,0:qy-1] = dens.d[0:qx-1,0:qy-1] + \
                   dtdx*(flux_x[0:qx-1,0:qy-1] - flux_x[1:qx,0:qy-1]) + \
                   dtdy*(flux_y[0:qx-1,0:qy-1] - flux_y[0:qx-1,1:qy])


    def dovis(self):
        """
        Do runtime visualization.
        """
        plt.clf()

        dens = self.cc_data.get_var("density")

        myg = self.cc_data.grid

        plt.imshow(np.transpose(dens.v()),
                   interpolation="nearest", origin="lower",
                   extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("density")

        plt.colorbar()

        plt.figtext(0.05,0.0125, "t = %10.5f" % self.cc_data.t)

        plt.draw()


