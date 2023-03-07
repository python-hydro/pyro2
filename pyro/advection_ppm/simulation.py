import importlib

import pyro.mesh.array_indexer as ai
import pyro.mesh.patch as patch
import pyro.particles.particles as particles
from pyro.simulation_null import NullSimulation, bc_setup, grid_setup
from pyro.mesh.reconstruction import ppm_reconstruction

class Simulation(NullSimulation):

    def initialize(self):
        """
        Initialize the grid and the initial conditions.
        """
        my_grid = grid_setup(self.rp, ng = 3)
        my_data = patch.CenterCellData2D(my_grid)
        bc = bc_setup(self.rp)[0]

        my_data.register_var("density", bc)
        my_data.create()

        if self.rp.get_param("particles.do_particles") == 1:
            n_particles = self.rp.get_param("particles.n_particles")
            particle_generator = self.rp.get_param("particles.particle_generator")
            self.particles = particles.Particles(self.cc_data, bc, n_particles, particle_generator)

        # now set the initial conditions for the problem
        problem = importlib.import_module(f"pyro.advection_ppm.problems.{self.problem_name}")
        problem.init_data(self.cc_data, self.rp)

    def ppm_step(self):
        """ We compute the timestep evolution, from the reconstruction method"""
