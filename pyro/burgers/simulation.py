import matplotlib.pyplot as plt
import numpy as np

from pyro.burgers import burgers_interface
from pyro.mesh import patch, reconstruction
from pyro.particles import particles
from pyro.simulation_null import NullSimulation, bc_setup, grid_setup
from pyro.util import plot_tools


class Simulation(NullSimulation):

    def initialize(self):
        """
        Initialize the grid and variables for advection and set the initial
        conditions for the chosen problem.
        """

        # create grid, self.rp contains mesh.nx and mesh.ny
        my_grid = grid_setup(self.rp, ng=4)

        # create the variables
        my_data = patch.CellCenterData2d(my_grid)

        # outputs: bc, bc_xodd and bc_yodd for reflection boundary cond
        bc = bc_setup(self.rp)[0]

        # register variables in the data
        # burgers equation advects velocity

        my_data.register_var("x-velocity", bc)
        my_data.register_var("y-velocity", bc)
        my_data.create()

        # holds various data, like time and registered variable.
        self.cc_data = my_data

        if self.rp.get_param("particles.do_particles") == 1:
            n_particles = self.rp.get_param("particles.n_particles")
            particle_generator = self.rp.get_param("particles.particle_generator")
            self.particles = particles.Particles(self.cc_data, bc, n_particles, particle_generator)

        # now set the initial conditions for the problem
        self.problem_func(self.cc_data, self.rp)

    def method_compute_timestep(self):
        """
        The timestep() function computes the advective timestep
        (CFL) constraint.  The CFL constraint says that information
        cannot propagate further than one zone per timestep.

        We use the driver.cfl parameter to control what fraction of the CFL
        step we actually take.
        """

        cfl = self.rp.get_param("driver.cfl")

        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        # the timestep is min(dx/|u|, dy|v|)

        xtmp = self.cc_data.grid.dx / max(abs(u).max(), self.SMALL)
        ytmp = self.cc_data.grid.dy / max(abs(v).max(), self.SMALL)

        self.dt = cfl * min(xtmp, ytmp)

    def evolve(self):
        """
        Evolve the burgers equation through one timestep.
        """

        myg = self.cc_data.grid

        dtdx = self.dt/myg.dx
        dtdy = self.dt/myg.dy

        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        # --------------------------------------------------------------------------
        # monotonized central differences
        # --------------------------------------------------------------------------

        limiter = self.rp.get_param("advection.limiter")

        # Give da/dx and da/dy using input: (state, grid, direction, limiter)

        ldelta_ux = reconstruction.limit(u, myg, 1, limiter)
        ldelta_uy = reconstruction.limit(u, myg, 2, limiter)
        ldelta_vx = reconstruction.limit(v, myg, 1, limiter)
        ldelta_vy = reconstruction.limit(v, myg, 2, limiter)

        # Get u, v fluxes

        u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr = burgers_interface.get_interface_states(myg, self.dt,
                                                                                                u, v,
                                                                                                ldelta_ux, ldelta_vx,
                                                                                                ldelta_uy, ldelta_vy)

        u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr = burgers_interface.apply_transverse_corrections(myg, self.dt,
                                                                                                u_xl, u_xr,
                                                                                                u_yl, u_yr,
                                                                                                v_xl, v_xr,
                                                                                                v_yl, v_yr)

        u_flux_x, u_flux_y, v_flux_x, v_flux_y = burgers_interface.construct_unsplit_fluxes(myg,
                                                                                            u_xl, u_xr,
                                                                                            u_yl, u_yr,
                                                                                            v_xl, v_xr,
                                                                                            v_yl, v_yr)

        """
        do the differencing for the fluxes now.  Here, we use slices so we
        avoid slow loops in python.  This is equivalent to:

        myPatch.data[i,j] = myPatch.data[i,j] + \
                               dtdx*(flux_x[i,j] - flux_x[i+1,j]) + \
                               dtdy*(flux_y[i,j] - flux_y[i,j+1])
        """

        u.v()[:, :] = u.v() + dtdx*(u_flux_x.v() - u_flux_x.ip(1)) + \
                              dtdy*(u_flux_y.v() - u_flux_y.jp(1))

        v.v()[:, :] = v.v() + dtdx*(v_flux_x.v() - v_flux_x.ip(1)) + \
                              dtdy*(v_flux_y.v() - v_flux_y.jp(1))

        if self.particles is not None:

            u2d = u
            v2d = v

            self.particles.update_particles(self.dt, u2d, v2d)

        # increment the time
        self.cc_data.t += self.dt
        self.n += 1

    def dovis(self):
        """
        Do runtime visualization
        """
        plt.clf()

        plt.rc("font", size=10)

        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        myg = self.cc_data.grid
        _, axes, _ = plot_tools.setup_axes(myg, 2)

        fields = [u, v]
        field_names = ["u", "v"]

        for n in range(2):
            ax = axes[n]

            f = fields[n]
            img = ax.imshow(np.transpose(f.v()),
                            interpolation="nearest", origin="lower",
                            extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax], cmap=self.cm)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(field_names[n])

            cb = axes.cbar_axes[n].colorbar(img)
            cb.solids.set_rasterized(True)
            cb.solids.set_edgecolor("face")

            if self.particles is not None:

                particle_positions = self.particles.get_positions()
                # dye particles
                colors = self.particles.get_init_positions()[:, 0]

                # plot particles
                ax.scatter(particle_positions[:, 0],
                           particle_positions[:, 1], s=8, c=colors, cmap="Greys")
                ax.set_xlim([myg.xmin, myg.xmax])
                ax.set_ylim([myg.ymin, myg.ymax])

        plt.figtext(0.05, 0.0125, f"t = {self.cc_data.t:10.5f}")

        plt.pause(0.001)
        plt.draw()
