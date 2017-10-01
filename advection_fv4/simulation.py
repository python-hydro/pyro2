from __future__ import print_function

import importlib
import advection_rk
import mesh.integration as integration
import mesh.array_indexer as ai
import mesh.fv as fv
from simulation_null import grid_setup, bc_setup

class Simulation(advection_rk.Simulation):

    def initialize(self):
        """
        Initialize the grid and variables for advection and set the initial
        conditions for the chosen problem.
        """

        my_grid = grid_setup(self.rp, ng=4)

        # create the variables
        my_data = fv.FV(my_grid)
        bc = bc_setup(self.rp)[0]
        my_data.register_var("density", bc)
        my_data.create()

        self.cc_data = my_data

        # now set the initial conditions for the problem
        problem = importlib.import_module("advection.problems.{}".format(self.problem_name))
        problem.init_data(self.cc_data, self.rp)
