import pyro.advection_fv4.fluxes as flx
import pyro.mesh.array_indexer as ai
from pyro import advection_rk
from pyro.mesh import fv
from pyro.particles import particles
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
        self.problem_func(self.cc_data, self.rp)

    def substep(self, myd):
        """
        take a single substep in the RK timestepping starting with the
        conservative state defined as part of myd
        """

        # this is identical to the RK version, but we need to
        # dupe it here to pull in the correct fluxes

        myg = myd.grid

        k = myg.scratch_array()

        flux_x, flux_y = flx.fluxes(myd, self.rp)

        F_x = ai.ArrayIndexer(d=flux_x, grid=myg)
        F_y = ai.ArrayIndexer(d=flux_y, grid=myg)

        k.v()[:, :] = \
            (F_x.v() - F_x.ip(1))/myg.dx + \
            (F_y.v() - F_y.jp(1))/myg.dy

        return k

    def preevolve(self):
        """Since we are 4th order accurate we need to make sure that we
        initialized with accurate zone-averages, so the preevolve for
        this solver assumes that the initialization was done to
        cell-centers and converts it to cell-averages."""

        # we just initialized cell-centers, but we need to store averages
        for var in self.cc_data.names:
            self.cc_data.from_centers(var)
