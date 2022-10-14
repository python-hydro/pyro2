import importlib

import matplotlib
import numpy as np

try:
    matplotlib.rcParams['mpl_toolkits.legacy_colorbar'] = False
except KeyError:
    pass
import matplotlib.pyplot as plt

import pyro.advection_nonuniform.advective_fluxes as flx
import pyro.mesh.patch as patch
import pyro.particles.particles as particles
import pyro.util.plot_tools as plot_tools
from pyro.simulation_null import NullSimulation, bc_setup, grid_setup


class Simulation(NullSimulation):

    def initialize(self):
        """
        Initialize the grid and variables for advection and set the initial
        conditions for the chosen problem.
        """
        def shift(velocity):
            """
            Computes the direction of shift for each node for upwind
            discretization based on sign of veclocity
            """
            shift_vel = np.sign(velocity)
            shift_vel[np.where(shift_vel <= 0)] = 0
            shift_vel[np.where(shift_vel > 0)] = -1
            return shift_vel

        my_grid = grid_setup(self.rp, ng=4)

        # create the variables
        bc, bc_xodd, bc_yodd = bc_setup(self.rp)

        my_data = patch.CellCenterData2d(my_grid)

        # velocities
        my_data.register_var("x-velocity", bc_xodd)
        my_data.register_var("y-velocity", bc_yodd)

        # shift
        my_data.register_var("x-shift", bc_xodd)
        my_data.register_var("y-shift", bc_yodd)

        # density
        my_data.register_var("density", bc)

        my_data.create()

        self.cc_data = my_data

        if self.rp.get_param("particles.do_particles") == 1:
            n_particles = self.rp.get_param("particles.n_particles")
            particle_generator = self.rp.get_param("particles.particle_generator")
            self.particles = particles.Particles(self.cc_data, bc, n_particles, particle_generator)

        # now set the initial conditions for the problem
        problem = importlib.import_module(f"pyro.advection_nonuniform.problems.{self.problem_name}")
        problem.init_data(self.cc_data, self.rp)

        # compute the required shift for each node using corresponding velocity at the node
        shx = self.cc_data.get_var("x-shift")
        shx[:, :] = shift(self.cc_data.get_var("x-velocity"))
        shy = self.cc_data.get_var("y-shift")
        shy[:, :] = shift(self.cc_data.get_var("y-velocity"))

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
        xtmp = self.cc_data.grid.dx/np.amax(np.fabs(u))
        ytmp = self.cc_data.grid.dy/np.amax(np.fabs(v))

        self.dt = cfl*float(min(xtmp, ytmp))

    def evolve(self):
        """
        Evolve the linear advection equation through one timestep.  We only
        consider the "density" variable in the CellCenterData2d object that
        is part of the Simulation.
        """
        myd = self.cc_data

        dtdx = self.dt/myd.grid.dx
        dtdy = self.dt/myd.grid.dy

        flux_x, flux_y = flx.unsplit_fluxes(myd, self.rp, self.dt, "density")

        """
        do the differencing for the fluxes now.  Here, we use slices so we
        avoid slow loops in python.  This is equivalent to:

        myPatch.data[i,j] = myPatch.data[i,j] + \
                               dtdx*(flux_x[i,j] - flux_x[i+1,j]) + \
                               dtdy*(flux_y[i,j] - flux_y[i,j+1])
        """
        dens = myd.get_var("density")

        dens.v()[:, :] = dens.v() + dtdx*(flux_x.v() - flux_x.ip(1)) + \
                                    dtdy*(flux_y.v() - flux_y.jp(1))

        if self.particles is not None:
            u = myd.get_var("x-velocity")
            v = myd.get_var("y-velocity")

            self.particles.update_particles(self.dt, u, v)

        # increment the time
        myd.t += self.dt
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
