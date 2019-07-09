from __future__ import print_function

import importlib

import numpy as np
import matplotlib.pyplot as plt

import sph.compute as compute
import mesh.boundary as bnd
from simulation_null import NullSimulation
import util.plot_tools as plot_tools
from util import msg
import sph.particles as particles
from scipy.interpolate import interp2d

def domain_setup(rp, ng=1):

    try:
        xmin = rp.get_param("mesh.xmin")
    except KeyError:
        xmin = 0.0
        msg.warning("mesh.xmin not set, defaulting to 0.0")

    try:
        xmax = rp.get_param("mesh.xmax")
    except KeyError:
        xmax = 1.0
        msg.warning("mesh.xmax not set, defaulting to 1.0")

    try:
        ymin = rp.get_param("mesh.ymin")
    except KeyError:
        ymin = 0.0
        msg.warning("mesh.ymin not set, defaulting to 0.0")

    try:
        ymax = rp.get_param("mesh.ymax")
    except KeyError:
        ymax = 1.0
        msg.warning("mesh.ynax not set, defaulting to 1.0")

    my_domain = particles.Domain2d(xmin=xmin, xmax=xmax,
                           ymin=ymin, ymax=ymax)
    return my_domain

def bc_setup(rp):

    # first figure out the BCs
    try:
        xlb_type = rp.get_param("mesh.xlboundary")
    except KeyError:
        xlb_type = "periodic"
        msg.warning("mesh.xlboundary is not set, defaulting to periodic")

    try:
        xrb_type = rp.get_param("mesh.xrboundary")
    except KeyError:
        xrb_type = "periodic"
        msg.warning("mesh.xrboundary is not set, defaulting to periodic")

    try:
        ylb_type = rp.get_param("mesh.ylboundary")
    except KeyError:
        ylb_type = "periodic"
        msg.warning("mesh.ylboundary is not set, defaulting to periodic")

    try:
        yrb_type = rp.get_param("mesh.yrboundary")
    except KeyError:
        yrb_type = "periodic"
        msg.warning("mesh.yrboundary is not set, defaulting to periodic")

    bc = bnd.BC(xlb=xlb_type, xrb=xrb_type,
                ylb=ylb_type, yrb=yrb_type)

    return bc


class Simulation(NullSimulation):
    """The main simulation class for the corner transport upwind
    sph hydrodynamics solver

    """

    def __init__(self, solver_name, problem_name, rp, timers=None, data_class=particles.ParticleData2d):

        return super().__init__(solver_name, problem_name, rp, timers, data_class)

    def initialize(self, extra_vars=None, ng=4):
        """
        Initialize the grid and variables for sph flow and set
        the initial conditions for the chosen problem.
        """
        my_domain = domain_setup(self.rp)
        bc = bc_setup(self.rp)

        my_data = self.data_class(my_domain, self.rp, bc)

        # are we dealing with solid boundaries? we'll use these for
        # the Riemann solver
        self.solid = bnd.bc_is_solid(bc)

        # density and energy
        my_data.register_var("mass")
        my_data.register_var("density")
        my_data.register_var("x-position")
        my_data.register_var("y-position")
        my_data.register_var("x-velocity")
        my_data.register_var("y-velocity")
        my_data.register_var("half-x-velocity")
        my_data.register_var("half-y-velocity")
        my_data.register_var("x-acceleration")
        my_data.register_var("y-acceleration")

        # store the gravitational acceration g as an auxillary quantity
        # so we can have a
        # self-contained object stored in output files to make plots.
        # store grav because we'll need that in some BCs
        my_data.set_aux("g", self.rp.get_param("sph.grav"))
        my_data.set_aux("h", self.rp.get_param("sph.particle_size"))
        my_data.set_aux("rho0", self.rp.get_param("sph.rho0"))
        my_data.set_aux("k", self.rp.get_param("sph.k"))
        my_data.set_aux("mu", self.rp.get_param("sph.mu"))

        my_data.create()

        self.cc_data = my_data

        self.ivars = particles.Variables(my_data)

        # initial conditions for the problem
        problem = importlib.import_module("{}.problems.{}".format(
            self.solver_name, self.problem_name))
        problem.init_data(self.cc_data, self.rp)

        self.cc_data.normalize_mass()

        self.compute_timestep()

        if self.verbose > 0:
            print(my_data)

    def compute_timestep(self):
        """
        The timestep function computes the advective timestep (CFL)
        constraint.  The CFL constraint says that information cannot
        propagate further than one zone per timestep.

        We use the driver.cfl parameter to control what fraction of the
        CFL step we actually take.
        """

        cfl = self.rp.get_param("driver.cfl")

        dx = self.cc_data.domain.xmax - self.cc_data.domain.xmin
        dy = self.cc_data.domain.ymax - self.cc_data.domain.ymin

        max_u = np.max(np.abs(self.cc_data.get_var("x-velocity")))
        max_v = np.max(np.abs(self.cc_data.get_var("y-velocity")))

        self.dt = np.min((self.rp.get_param("sph.dt"), cfl*dx/max_u, cfl*dy/max_v))

    def preevolve(self):
        """
        Evolve the equations of sph hydrodynamics through a
        timestep dt.
        """

        tm_evolve = self.tc.timer("preevolve")
        tm_evolve.begin()

        U = self.cc_data.data
        ivars = particles.Variables(self.cc_data)

        # compute the acceleration
        compute.compute_acceleration(self.cc_data, ivars)

        # update velocity at half timestep
        U[:, ivars.iuh:ivars.ivh+1] = U[:, ivars.iu:ivars.iv+1] + U[:, ivars.iax:ivars.iay+1] * self.dt * 0.5

        # update velocity
        U[:, ivars.iu:ivars.iv+1] += U[:, ivars.iax:ivars.iay+1] * self.dt

        # update position
        U[:, ivars.ix:ivars.iy+1] += U[:, ivars.iuh:ivars.ivh+1] * self.dt

        tm_evolve.end()

        self.cc_data.fill_BC_all()

    def evolve(self):
        """
        Evolve the equations of sph hydrodynamics through a
        timestep dt.
        """

        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()

        U = self.cc_data.data
        ivars = particles.Variables(self.cc_data)

        # compute the acceleration
        compute.compute_acceleration(self.cc_data, ivars)

        # update velocity at half timestep
        U[:, ivars.iuh:ivars.ivh+1] += U[:, ivars.iax:ivars.iay+1] * self.dt

        # update velocity
        U[:, ivars.iu:ivars.iv+1] = U[:, ivars.iuh:ivars.ivh+1] + U[:, ivars.iax:ivars.iay+1] * self.dt * 0.5

        # update position
        U[:, ivars.ix:ivars.iy+1] += U[:, ivars.iuh:ivars.ivh+1] * self.dt

        # increment the time
        self.cc_data.t += self.dt
        self.n += 1

        self.cc_data.fill_BC_all()

        tm_evolve.end()

    def dovis(self):
        """
        Do runtime visualization.
        """

        plt.clf()

        plt.rc("font", size=10)

        _, axes, cbar_title = plot_tools.setup_axes(self.cc_data.domain, 1)

        myd = self.cc_data
        # ivars = particles.Variables(self.cc_data)

        xs = myd.get_var("x-position")
        ys = myd.get_var("y-position")

        # compute.compute_density(self.cc_data, ivars)
        #
        # dens = myd.get_var("density")
        #
        # # f = interp2d(xs, ys, dens, kind='cubic')
        #
        # X = np.linspace(self.cc_data.domain.xmin, self.cc_data.domain.xmax, 100)
        # Y = np.linspace(self.cc_data.domain.ymin, self.cc_data.domain.ymax, 100)

        # interp_dens = f(X, Y)

        # print(interp_dens)

        ax = axes[0]

        # ax.imshow(np.transpose(interp_dens))
        ax.scatter(xs, ys)#, c=np.array(range(self.cc_data.np)))
        ax.set_xlim([self.cc_data.domain.xmin, self.cc_data.domain.xmax])
        ax.set_ylim([self.cc_data.domain.ymin, self.cc_data.domain.ymax])

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # # access g from the cc_data object so we can use dovis
        # # outside of a running simulation.
        # g = self.cc_data.get_aux("g")
        #
        #
        # h = q[:, :, ivars.ih]
        # u = q[:, :, ivars.iu]
        # v = q[:, :, ivars.iv]
        # fuel = q[:, :, ivars.ix]
        #
        # magvel = np.sqrt(u**2 + v**2)
        #
        # myg = self.cc_data.grid
        #
        # vort = myg.scratch_array()
        #
        # dv = 0.5*(v.ip(1) - v.ip(-1))/myg.dx
        # du = 0.5*(u.jp(1) - u.jp(-1))/myg.dy
        #
        # vort.v()[:, :] = dv - du
        #
        # fields = [h, magvel, fuel, vort]
        # field_names = [r"$h$", r"$|U|$", r"$X$", r"$\nabla\times U$"]
        #
        # _, axes, cbar_title = plot_tools.setup_axes(myg, len(fields))
        #
        # for n, ax in enumerate(axes):
        #     v = fields[n]
        #
        #     img = ax.imshow(np.transpose(v.v()),
        #                     interpolation="nearest", origin="lower",
        #                     extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
        #                     cmap=self.cm)
        #
        #     ax.set_xlabel("x")
        #     ax.set_ylabel("y")
        #
        #     # needed for PDF rendering
        #     cb = axes.cbar_axes[n].colorbar(img)
        #     cb.solids.set_rasterized(True)
        #     cb.solids.set_edgecolor("face")
        #
        #     if cbar_title:
        #         cb.ax.set_title(field_names[n])
        #     else:
        #         ax.set_title(field_names[n])
        #
        plt.figtext(0.05, 0.0125, "t = {:10.5g}".format(self.cc_data.t))

        plt.pause(0.0001)
        plt.draw()
