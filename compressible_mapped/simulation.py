from __future__ import print_function

import numpy as np
import importlib
import matplotlib.pyplot as plt

import mesh.integration as integration
import compressible
import compressible_rk
import compressible_mapped.fluxes as flx
from util import msg, plot_tools
import mesh.mapped as mapped
import mesh.boundary as bnd
import compressible.BC as BC
import compressible.eos as eos
from simulation_null import bc_setup
import particles.particles as particles
import compressible.derives as derives


def mapped_grid_setup(rp, map, ng=1):
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

    my_grid = mapped.MappedGrid2d(map, nx, ny,
                                  xmin=xmin, xmax=xmax,
                                  ymin=ymin, ymax=ymax, ng=ng)
    return my_grid

def mapped_bc_setup(rp):

    bnd.define_bc("mapped_periodic", BC.user, is_solid=False)
    bnd.define_bc("mapped_outflow", BC.user, is_solid=False)
    bnd.define_bc("mapped_neumann", BC.user, is_solid=False)
    bnd.define_bc("mapped_reflect", BC.user, is_solid=True)
    bnd.define_bc("mapped_reflect-even", BC.user, is_solid=True)
    bnd.define_bc("mapped_reflect-odd", BC.user, is_solid=True)
    bnd.define_bc("mapped_dirichlet", BC.user, is_solid=True)

    # first figure out the BCs
    try:
        xlb_type = "mapped_" + rp.get_param("mesh.xlboundary")
    except KeyError:
        xlb_type = "mapped_periodic"
        msg.warning("mesh.xlboundary is not set, defaulting to mapped_periodic")

    try:
        xrb_type = "mapped_" + rp.get_param("mesh.xrboundary")
    except KeyError:
        xrb_type = "mapped_periodic"
        msg.warning("mesh.xrboundary is not set, defaulting to mapped_periodic")

    try:
        ylb_type = "mapped_" + rp.get_param("mesh.ylboundary")
    except KeyError:
        ylb_type = "mapped_periodic"
        msg.warning("mesh.ylboundary is not set, defaulting to mapped_periodic")

    try:
        yrb_type = "mapped_" + rp.get_param("mesh.yrboundary")
    except KeyError:
        yrb_type = "mapped_periodic"
        msg.warning("mesh.yrboundary is not set, defaulting to mapped_periodic")

    bc = bnd.BC(xlb=xlb_type, xrb=xrb_type,
                ylb=ylb_type, yrb=yrb_type)

    # if we are reflecting, we need odd reflection in the normal
    # directions for the velocity
    bc_xodd = bnd.BC(xlb=xlb_type, xrb=xrb_type,
                     ylb=ylb_type, yrb=yrb_type,
                     odd_reflect_dir="x")

    bc_yodd = bnd.BC(xlb=xlb_type, xrb=xrb_type,
                     ylb=ylb_type, yrb=yrb_type,
                     odd_reflect_dir="y")

    return bc, bc_xodd, bc_yodd


class Simulation(compressible_rk.Simulation):
    """The main simulation class for the method of lines compressible
    hydrodynamics solver"""

    def __init__(self, solver_name, problem_name, rp, timers=None):

        super().__init__(solver_name, problem_name, rp, timers=timers,
                         data_class=mapped.MappedCellCenterData2d)

    def initialize(self, extra_vars=None, ng=4):
        """
        Initialize the grid and variables for compressible flow and set
        the initial conditions for the chosen problem.
        """
        problem = importlib.import_module("{}.problems.{}".format(
            self.solver_name, self.problem_name))

        my_grid = mapped_grid_setup(self.rp, problem.sym_map, ng=ng)
        my_data = self.data_class(my_grid)

        # # define solver specific boundary condition routines
        # bnd.define_bc("hse", BC.user, is_solid=False)
        # # for double mach reflection problem
        # bnd.define_bc("ramp", BC.user, is_solid=False)

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

        # make rotation matrices
        my_data.make_rotation_matrices(self.ivars)

        # derived variables
        self.cc_data.add_derived(derives.derive_primitives)

        # initial conditions for the problem
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
        b = 3
        for n in range(self.ivars.nvar):
            k.v(n=n, buf=b)[:, :] = \
               (flux_xp.v(n=n, buf=b) + flux_xm.ip(1, n=n, buf=b)) / (myg.dx * myg.kappa.v(buf=b)) + \
               (flux_yp.v(n=n, buf=b) + flux_ym.jp(1, n=n, buf=b)) / \
                (myg.dy * myg.kappa.v(buf=b))

        return -k

    def evolve(self):
        """
        Evolve the equations of compressible hydrodynamics through a
        timestep dt.
        """

        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()

        myd = self.cc_data

        method = self.rp.get_param("compressible.temporal_method")

        rk = integration.RKIntegrator(myd.t, self.dt, method=method)
        rk.set_start(myd)

        for s in range(rk.nstages()):
            ytmp = rk.get_stage_start(
                s, clone_function=mapped.mapped_cell_center_data_clone)
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

    def dovis(self):
        """
        Do runtime visualization.
        """

        plt.clf()

        plt.rc("font", size=10)

        # we do this even though ivars is in self, so this works when
        # we are plotting from a file
        ivars = compressible.Variables(self.cc_data)

        # access gamma from the cc_data object so we can use dovis
        # outside of a running simulation.
        gamma = self.cc_data.get_aux("gamma")

        q = compressible.cons_to_prim(
            self.cc_data.data, gamma, ivars, self.cc_data.grid)

        rho = q[:, :, ivars.irho]
        u = q[:, :, ivars.iu]
        v = q[:, :, ivars.iv]
        p = q[:, :, ivars.ip]
        e = eos.rhoe(gamma, p) / rho

        magvel = np.sqrt(u**2 + v**2)

        myg = self.cc_data.grid

        fields = [rho, magvel, p, e]
        field_names = [r"$\rho$", r"U", "p", "e"]

        _, axes, cbar_title = plot_tools.setup_axes(myg, len(fields))

        X, Y = myg.physical_coords()
        X = X[myg.ng:-myg.ng, myg.ng:-myg.ng]
        Y = Y[myg.ng:-myg.ng, myg.ng:-myg.ng]

        for n, ax in enumerate(axes):
            v = fields[n]

            img = ax.pcolormesh(X, Y, v.v(), cmap=self.cm)

            ax.set_xlabel("x")
            ax.set_ylabel("y")

            # needed for PDF rendering
            cb = axes.cbar_axes[n].colorbar(img)
            cb.solids.set_rasterized(True)
            cb.solids.set_edgecolor("face")

            if cbar_title:
                cb.ax.set_title(field_names[n])
            else:
                ax.set_title(field_names[n])

        if self.particles is not None:
            ax = axes[0]
            particle_positions = self.particles.get_positions()
            # dye particles
            colors = self.particles.get_init_positions()[:, 0]

            # plot particles
            ax.scatter(particle_positions[:, 0],
                       particle_positions[:, 1], s=5, c=colors, alpha=0.8, cmap="Greys")
            ax.set_xlim([myg.xmin, myg.xmax])
            ax.set_ylim([myg.ymin, myg.ymax])

        plt.figtext(0.05, 0.0125, "t = {:10.5g}".format(self.cc_data.t))

        plt.pause(0.001)
        plt.draw()
