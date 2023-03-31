import importlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyro.mesh.patch as patch
import pyro.particles.particles as particles
import pyro.util.plot_tools as plot_tools
from pyro.simulation_null import NullSimulation, bc_setup, grid_setup
import pyro.advection_ppm.fluxes as flx


class Simulation(NullSimulation):

    def initialize(self):
        """
        Initialize the grid and the initial conditions.
        """

        my_grid = grid_setup(self.rp, ng=4)
        my_data = patch.CellCenterData2d(my_grid)
        bc = bc_setup(self.rp)[0]

        my_data.register_var("density", bc)
        my_data.create()

        self.cc_data = my_data

        if self.rp.get_param("particles.do_particles") == 1:
            n_particles = self.rp.get_param("particles.n_particles")
            particle_generator = self.rp.get_param("particles.particle_generator")
            self.particles = particles.Particles(self.cc_data, bc, n_particles, particle_generator)

        # now set the initial conditions for the problem
        problem = importlib.import_module(f"pyro.advection_ppm.problems.{self.problem_name}")
        problem.init_data(self.cc_data, self.rp)

    def method_compute_timestep(self):
        """
        Compute the advective timestep (CFL) constraint.  We use the
        driver.cfl parameter to control what fraction of the CFL
        step we actually take.
        """

        cfl = self.rp.get_param("driver.cfl")

        u = self.rp.get_param("advection.u")
        v = self.rp.get_param("advection.v")

        # the timestep is min(dx/|u|, dy/|v|)
        xtmp = self.cc_data.grid.dx/max(abs(u), self.SMALL)
        ytmp = self.cc_data.grid.dy/max(abs(v), self.SMALL)

        self.dt = cfl*min(xtmp, ytmp)

    def evolve(self):
        """
        Evolve the linear advection equation through one timestep.  We only
        consider the "density" variable in the CellCenterData2d object that
        is part of the Simulation.
        """

        dtdx = self.dt/self.cc_data.grid.dx
        dtdy = self.dt/self.cc_data.grid.dy

        Fx, Fy = flx.ctu_unsplit_fluxes(self.cc_data, self.rp, self.dt, "density")

        dens = self.cc_data.get_var("density")

        dens.v()[:, :] = dens.v() + dtdx*(Fx.v() - Fx.ip(1)) + \
                                    dtdy*(Fy.v() - Fy.jp(1))

        if self.particles is not None:
            myg = self.cc_data.grid
            u = self.rp.get_param("advection.u")
            v = self.rp.get_param("advection.v")

            u2d = myg.scratch_array() + u
            v2d = myg.scratch_array() + v

            self.particles.update_particles(self.dt, u2d, v2d)

        # increment the time
        self.cc_data.t += self.dt
        self.n += 1

    def dovis(self):
        """
        Do runtime visualization.
        """
        plt.clf()

        dens = self.cc_data.get_var("density")

        myg = self.cc_data.grid

        _, axes, cbar_title = plot_tools.setup_axes(myg, 1)

        # plot density
        ax = axes[0]
        img = ax.imshow(np.transpose(dens.v()),
                   interpolation="nearest", origin="lower",
                   extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
                   cmap=self.cm)

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # needed for PDF rendering
        cb = axes.cbar_axes[0].colorbar(img)
        cb.formatter = matplotlib.ticker.FormatStrFormatter("")
        cb.solids.set_rasterized(True)
        cb.solids.set_edgecolor("face")

        plt.title("density")

        if self.particles is not None:
            particle_positions = self.particles.get_positions()

            # dye particles
            colors = self.particles.get_init_positions()[:, 0]

            # plot particles
            ax.scatter(particle_positions[:, 0],
                particle_positions[:, 1], c=colors, cmap="Greys")
            ax.set_xlim([myg.xmin, myg.xmax])
            ax.set_ylim([myg.ymin, myg.ymax])

        plt.figtext(0.05, 0.0125, f"t = {self.cc_data.t:10.5f}")

        plt.pause(0.001)
        plt.draw()
