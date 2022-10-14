import importlib

import pyro.advection_fv4.fluxes as flx
import pyro.advection_rk as advection_rk
import pyro.mesh.array_indexer as ai
import pyro.mesh.fv as fv
import pyro.particles.particles as particles
from pyro.simulation_null import bc_setup, grid_setup


class Simulation(advection_rk.Simulation):

    def initialize(self):
        """
        Initialize the grid and variables for advection and set the initial
        conditions for the chosen problem.
        """

        my_grid = grid_setup(self.rp, ng=4)

        # create the variables
        my_data = fv.FV2d(my_grid)
        bc = bc_setup(self.rp)[0]
        my_data.register_var("density", bc)
        my_data.create()

        self.cc_data = my_data

        if self.rp.get_param("particles.do_particles") == 1:
            n_particles = self.rp.get_param("particles.n_particles")
            particle_generator = self.rp.get_param("particles.particle_generator")
            self.particles = particles.Particles(self.cc_data, bc, n_particles, particle_generator)

        # now set the initial conditions for the problem
        problem = importlib.import_module(f"pyro.advection_fv4.problems.{self.problem_name}")
        problem.init_data(self.cc_data, self.rp)

    def substep(self, myd):
        """
        take a single substep in the RK timestepping starting with the
        conservative state defined as part of myd
        """

        # this is identical to the RK version, but we need to
        # dupe it here to pull in the correct fluxes

        myg = myd.grid

        k = myg.scratch_array()

        flux_x, flux_y = flx.fluxes(myd, self.rp, self.dt)

        F_x = ai.ArrayIndexer(d=flux_x, grid=myg)
        F_y = ai.ArrayIndexer(d=flux_y, grid=myg)

        k.v()[:, :] = \
            (F_x.v() - F_x.ip(1))/myg.dx + \
            (F_y.v() - F_y.jp(1))/myg.dy

        return k
