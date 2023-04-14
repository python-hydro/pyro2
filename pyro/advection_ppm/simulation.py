import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pyro import advection
import pyro.advection_ppm.fluxes as flx
from pyro.util import plot_tools


class Simulation(advection.Simulation):

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

        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()

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

        tm_evolve.end()

    def dovis(self):
        """
        Do runtime visualization.
        """
        plt.clf()

        dens = self.cc_data.get_var("density")

        myg = self.cc_data.grid

        _, axes, _ = plot_tools.setup_axes(myg, 1)

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
