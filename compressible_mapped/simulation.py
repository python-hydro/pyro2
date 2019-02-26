from __future__ import print_function

import numpy as np
import importlib

import mesh.integration as integration
import compressible
import compressible_mapped.fluxes as flx
from util import msg
from mesh.mapped import MappedGrid2d
import mesh.boundary as bnd
import compressible.BC as BC
from simulation_null import bc_setup
import particles.particles as particles
import compressible.derives as derives


def mapped_grid_setup(rp, ng=1):
    nx = rp.get_param("mesh.nx")
    ny = rp.get_param("mesh.ny")

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

    my_grid = MappedGrid2d(nx, ny,
                           xmin=xmin, xmax=xmax,
                           ymin=ymin, ymax=ymax, ng=ng)
    return my_grid


class Simulation(compressible.Simulation):
    """The main simulation class for the method of lines compressible
    hydrodynamics solver"""

    def initialize(self, extra_vars=None, ng=4):
        """
        Initialize the grid and variables for compressible flow and set
        the initial conditions for the chosen problem.
        """
        my_grid = mapped_grid_setup(self.rp, ng=ng)
        my_data = self.data_class(my_grid)

        # define solver specific boundary condition routines
        bnd.define_bc("hse", BC.user, is_solid=False)
        # for double mach reflection problem
        bnd.define_bc("ramp", BC.user, is_solid=False)

        bc, bc_xodd, bc_yodd = bc_setup(self.rp)

        # are we dealing with solid boundaries? we'll use these for
        # the Riemann solver
        self.solid = bnd.bc_is_solid(bc)

        # density and energy
        my_data.register_var("density", bc)
        my_data.register_var("energy", bc)
        my_data.register_var("x-momentum", bc_xodd)
        my_data.register_var("y-momentum", bc_yodd)

        # any extras?
        if extra_vars is not None:
            for v in extra_vars:
                my_data.register_var(v, bc)

        # store the EOS gamma as an auxillary quantity so we can have a
        # self-contained object stored in output files to make plots.
        # store grav because we'll need that in some BCs
        my_data.set_aux("gamma", self.rp.get_param("eos.gamma"))
        my_data.set_aux("grav", self.rp.get_param("compressible.grav"))

        my_data.create()

        self.cc_data = my_data

        if self.rp.get_param("particles.do_particles") == 1:
            self.particles = particles.Particles(self.cc_data, bc, self.rp)

        # some auxillary data that we'll need to fill GC in, but isn't
        # really part of the main solution
        aux_data = self.data_class(my_grid)
        aux_data.register_var("ymom_src", bc_yodd)
        aux_data.register_var("E_src", bc)
        aux_data.create()
        self.aux_data = aux_data

        self.ivars = compressible.Variables(my_data)

        # derived variables
        self.cc_data.add_derived(derives.derive_primitives)

        # initial conditions for the problem
        problem = importlib.import_module("{}.problems.{}".format(
            self.solver_name, self.problem_name))
        problem.init_data(self.cc_data, self.rp)

        if self.verbose > 0:
            print(my_data)

    def substep(self, myd):
        """
        take a single substep in the RK timestepping starting with the
        conservative state defined as part of myd
        """

        myg = myd.grid
        k = myg.scratch_array(nvar=self.ivars.nvar)

        flux_xp, flux_xm, flux_yp, flux_ym = flx.fluxes(myd, self.rp,
                                                        self.ivars, self.solid, self.tc)

        # flux_x, _, flux_y, _ = flx.fluxes(myd, self.rp,
        #                             self.ivars, self.solid, self.tc)

        # for n in range(self.ivars.nvar):
        #     k.v(n=n)[:, :] = \
        #        (flux_x.v(n=n) - flux_x.ip(1, n=n))/(myg.dx * myg.kappa) + \
        #        (flux_y.v(n=n) - flux_y.jp(1, n=n))/(myg.dy * myg.kappa)

        for n in range(self.ivars.nvar):
            k.v(n=n)[:, :] = \
               (flux_xp.v(n=n) + flux_xm.ip(1, n=n)) / (myg.dx * myg.kappa.v()) + \
               (flux_yp.v(n=n) + flux_ym.jp(1, n=n)) / (myg.dy * myg.kappa.v())

        return -k

    def method_compute_timestep(self):
        """
        The timestep function computes the advective timestep (CFL)
        constraint.  The CFL constraint says that information cannot
        propagate further than one zone per timestep.

        We use the driver.cfl parameter to control what fraction of the
        CFL step we actually take.
        """

        cfl = self.rp.get_param("driver.cfl")

        # get the variables we need
        u, v, cs = self.cc_data.get_var(["velocity", "soundspeed"])

        # the timestep is min(dx/(|u| + cs), dy/(|v| + cs))
        xtmp = (abs(u) + cs) / self.cc_data.grid.dx
        ytmp = (abs(v) + cs) / self.cc_data.grid.dy

        self.dt = cfl * float(np.min(1.0 / (xtmp + ytmp)))

    def evolve(self):
        """
        Evolve the equations of compressible hydrodynamics through a
        timestep dt.
        """

        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()

        myd = self.cc_data

        # k = self.substep(myd)

        # for n in range(self.ivars.nvar):
        #     myd.data.v(n=n)[:, :] -= k.v(n=n) * self.dt

        method = self.rp.get_param("compressible.temporal_method")

        rk = integration.RKIntegrator(myd.t, self.dt, method=method)
        rk.set_start(myd)

        for s in range(rk.nstages()):
            ytmp = rk.get_stage_start(s)
            ytmp.fill_BC_all()
            k = self.substep(ytmp)
            rk.store_increment(s, k)

        rk.compute_final_update()

        if self.particles is not None:
            self.particles.update_particles(self.dt)

        # increment the time
        myd.t += self.dt
        self.n += 1

        tm_evolve.end()
