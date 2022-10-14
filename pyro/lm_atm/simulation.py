import importlib

import matplotlib
import numpy as np

try:
    matplotlib.rcParams['mpl_toolkits.legacy_colorbar'] = False
except KeyError:
    pass
import matplotlib.pyplot as plt

import pyro.lm_atm.LM_atm_interface as lm_interface
import pyro.mesh.array_indexer as ai
import pyro.mesh.boundary as bnd
import pyro.mesh.patch as patch
import pyro.mesh.reconstruction as reconstruction
import pyro.multigrid.variable_coeff_MG as vcMG
from pyro.simulation_null import NullSimulation, bc_setup, grid_setup


class Basestate:
    def __init__(self, ny, ng=0):
        self.ny = ny
        self.ng = ng
        self.qy = ny + 2*ng

        self.d = np.zeros((self.qy), dtype=np.float64)

        self.jlo = ng
        self.jhi = ng+ny-1

    def v(self, buf=0):
        return self.d[self.jlo-buf:self.jhi+1+buf]

    def v2d(self, buf=0):
        return self.d[np.newaxis, self.jlo-buf:self.jhi+1+buf]

    def v2dp(self, shift, buf=0):
        return self.d[np.newaxis, self.jlo+shift-buf:self.jhi+1+shift+buf]

    def jp(self, shift, buf=0):
        return self.d[self.jlo-buf+shift:self.jhi+1+buf+shift]


class Simulation(NullSimulation):

    def __init__(self, solver_name, problem_name, rp, timers=None):

        NullSimulation.__init__(self, solver_name, problem_name, rp, timers=timers)

        self.base = {}
        self.aux_data = None

    def initialize(self):
        """
        Initialize the grid and variables for low Mach atmospheric flow
        and set the initial conditions for the chosen problem.
        """

        myg = grid_setup(self.rp, ng=4)

        bc_dens, bc_xodd, bc_yodd = bc_setup(self.rp)

        my_data = patch.CellCenterData2d(myg)

        my_data.register_var("density", bc_dens)
        my_data.register_var("x-velocity", bc_xodd)
        my_data.register_var("y-velocity", bc_yodd)

        # we'll keep the internal energy around just as a diagnostic
        my_data.register_var("eint", bc_dens)

        # phi -- used for the projections.  The boundary conditions
        # here depend on velocity.  At a wall or inflow, we already
        # have the velocity we want on the boundary, so we want
        # Neumann (dphi/dn = 0).  For outflow, we want Dirichlet (phi
        # = 0) -- this ensures that we do not introduce any tangental
        # acceleration.
        bcs = []
        for bc in [self.rp.get_param("mesh.xlboundary"),
                   self.rp.get_param("mesh.xrboundary"),
                   self.rp.get_param("mesh.ylboundary"),
                   self.rp.get_param("mesh.yrboundary")]:
            if bc == "periodic":
                bctype = "periodic"
            elif bc in ["reflect", "slipwall"]:
                bctype = "neumann"
            elif bc in ["outflow"]:
                bctype = "dirichlet"
            bcs.append(bctype)

        bc_phi = bnd.BC(xlb=bcs[0], xrb=bcs[1], ylb=bcs[2], yrb=bcs[3])

        my_data.register_var("phi-MAC", bc_phi)
        my_data.register_var("phi", bc_phi)

        # gradp -- used in the projection and interface states.  We'll do the
        # same BCs as density
        my_data.register_var("gradp_x", bc_dens)
        my_data.register_var("gradp_y", bc_dens)

        my_data.create()

        self.cc_data = my_data

        # some auxillary data that we'll need to fill GC in, but isn't
        # really part of the main solution
        aux_data = patch.CellCenterData2d(myg)

        aux_data.register_var("coeff", bc_dens)
        aux_data.register_var("source_y", bc_yodd)

        aux_data.create()
        self.aux_data = aux_data

        # we also need storage for the 1-d base state -- we'll store this
        # in the main class directly.
        self.base["rho0"] = Basestate(myg.ny, ng=myg.ng)
        self.base["p0"] = Basestate(myg.ny, ng=myg.ng)

        # now set the initial conditions for the problem
        problem = importlib.import_module(f"pyro.lm_atm.problems.{self.problem_name}")
        problem.init_data(self.cc_data, self.base, self.rp)

        # Construct beta_0
        gamma = self.rp.get_param("eos.gamma")
        self.base["beta0"] = Basestate(myg.ny, ng=myg.ng)
        self.base["beta0"].d[:] = self.base["p0"].d**(1.0/gamma)

        # we'll also need beta_0 on vertical edges -- on the domain edges,
        # just do piecewise constant
        self.base["beta0-edges"] = Basestate(myg.ny, ng=myg.ng)
        self.base["beta0-edges"].jp(1)[:] = \
            0.5*(self.base["beta0"].v() + self.base["beta0"].jp(1))
        self.base["beta0-edges"].d[myg.jlo] = self.base["beta0"].d[myg.jlo]
        self.base["beta0-edges"].d[myg.jhi+1] = self.base["beta0"].d[myg.jhi]

    def make_prime(self, a, a0):
        return a - a0.v2d(buf=a0.ng)

    def method_compute_timestep(self):
        """
        The timestep() function computes the advective timestep
        (CFL) constraint.  The CFL constraint says that information
        cannot propagate further than one zone per timestep.

        We use the driver.cfl parameter to control what fraction of the CFL
        step we actually take.
        """

        myg = self.cc_data.grid

        cfl = self.rp.get_param("driver.cfl")

        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        # the timestep is min(dx/|u|, dy|v|)
        xtmp = ytmp = 1.e33
        if not abs(u).max() == 0:
            xtmp = myg.dx/abs(u.v()).max()
        if not abs(v).max() == 0:
            ytmp = myg.dy/abs(v.v()).max()

        dt = cfl*min(xtmp, ytmp)

        # We need an alternate timestep that accounts for buoyancy, to
        # handle the case where the velocity is initially zero.
        rho = self.cc_data.get_var("density")
        rho0 = self.base["rho0"]
        rhoprime = self.make_prime(rho, rho0)

        g = self.rp.get_param("lm-atmosphere.grav")

        F_buoy = (abs(rhoprime*g).v()/rho.v()).max()

        dt_buoy = np.sqrt(2.0*myg.dx/F_buoy)

        self.dt = min(dt, dt_buoy)
        if self.verbose > 0:
            print(f"timestep is {dt}")

    def preevolve(self):
        """
        preevolve is called before we being the timestepping loop.  For
        the low Mach solver, this does an initial projection on the
        velocity field and then goes through the full evolution to get the
        value of phi.  The fluid state (rho, u, v) is then reset to values
        before this evolve.
        """

        self.in_preevolve = True

        myg = self.cc_data.grid

        rho = self.cc_data.get_var("density")
        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        self.cc_data.fill_BC("density")
        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")

        # 1. do the initial projection.  This makes sure that our original
        # velocity field satisties div U = 0

        # the coefficent for the elliptic equation is beta_0^2/rho
        coeff = 1/rho
        beta0 = self.base["beta0"]
        coeff.v()[:, :] = coeff.v()*beta0.v2d()**2

        # next create the multigrid object.  We defined phi with
        # the right BCs previously
        mg = vcMG.VarCoeffCCMG2d(myg.nx, myg.ny,
                                 xl_BC_type=self.cc_data.BCs["phi"].xlb,
                                 xr_BC_type=self.cc_data.BCs["phi"].xrb,
                                 yl_BC_type=self.cc_data.BCs["phi"].ylb,
                                 yr_BC_type=self.cc_data.BCs["phi"].yrb,
                                 xmin=myg.xmin, xmax=myg.xmax,
                                 ymin=myg.ymin, ymax=myg.ymax,
                                 coeffs=coeff,
                                 coeffs_bc=self.cc_data.BCs["density"],
                                 verbose=0)

        # first compute div{beta_0 U}
        div_beta_U = mg.soln_grid.scratch_array()

        # u/v are cell-centered, divU is cell-centered
        div_beta_U.v()[:, :] = \
            0.5*beta0.v2d()*(u.ip(1) - u.ip(-1))/myg.dx + \
            0.5*(beta0.v2dp(1)*v.jp(1) - beta0.v2dp(-1)*v.jp(-1))/myg.dy

        # solve D (beta_0^2/rho) G (phi/beta_0) = D( beta_0 U )

        # set the RHS to divU and solve
        mg.init_RHS(div_beta_U)
        mg.solve(rtol=1.e-10)

        # store the solution in our self.cc_data object -- include a single
        # ghostcell
        phi = self.cc_data.get_var("phi")
        phi[:, :] = mg.get_solution(grid=myg)

        # get the cell-centered gradient of phi and update the
        # velocities
        # FIXME: this update only needs to be done on the interior
        # cells -- not ghost cells
        gradp_x, gradp_y = mg.get_solution_gradient(grid=myg)

        coeff = 1.0/rho
        coeff.v()[:, :] = coeff.v()*beta0.v2d()

        u.v()[:, :] -= coeff.v()*gradp_x.v()
        v.v()[:, :] -= coeff.v()*gradp_y.v()

        # fill the ghostcells
        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")

        # 2. now get an approximation to gradp at n-1/2 by going through the
        # evolution.

        # store the current solution -- we'll restore it in a bit
        orig_data = patch.cell_center_data_clone(self.cc_data)

        # get the timestep
        self.method_compute_timestep()

        # evolve
        self.evolve()

        # update gradp_x and gradp_y in our main data object
        new_gp_x = self.cc_data.get_var("gradp_x")
        new_gp_y = self.cc_data.get_var("gradp_y")

        orig_gp_x = orig_data.get_var("gradp_x")
        orig_gp_y = orig_data.get_var("gradp_y")

        orig_gp_x[:, :] = new_gp_x[:, :]
        orig_gp_y[:, :] = new_gp_y[:, :]

        self.cc_data = orig_data

        if self.verbose > 0:
            print("done with the pre-evolution")

        self.in_preevolve = False

    def evolve(self):
        """
        Evolve the low Mach system through one timestep.
        """

        rho = self.cc_data.get_var("density")
        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        gradp_x = self.cc_data.get_var("gradp_x")
        gradp_y = self.cc_data.get_var("gradp_y")

        # note: the base state quantities do not have valid ghost cells
        beta0 = self.base["beta0"]
        beta0_edges = self.base["beta0-edges"]

        rho0 = self.base["rho0"]

        phi = self.cc_data.get_var("phi")

        myg = self.cc_data.grid

        # ---------------------------------------------------------------------
        # create the limited slopes of rho, u and v (in both directions)
        # ---------------------------------------------------------------------
        limiter = self.rp.get_param("lm-atmosphere.limiter")

        ldelta_rx = reconstruction.limit(rho, myg, 1, limiter)
        ldelta_ux = reconstruction.limit(u, myg, 1, limiter)
        ldelta_vx = reconstruction.limit(v, myg, 1, limiter)

        ldelta_ry = reconstruction.limit(rho, myg, 2, limiter)
        ldelta_uy = reconstruction.limit(u, myg, 2, limiter)
        ldelta_vy = reconstruction.limit(v, myg, 2, limiter)

        # ---------------------------------------------------------------------
        # get the advective velocities
        # ---------------------------------------------------------------------

        """
        the advective velocities are the normal velocity through each cell
        interface, and are defined on the cell edges, in a MAC type
        staggered form

                         n+1/2
                        v
                         i,j+1/2
                    +------+------+
                    |             |
            n+1/2   |             |   n+1/2
           u        +     U       +  u
            i-1/2,j |      i,j    |   i+1/2,j
                    |             |
                    +------+------+
                         n+1/2
                        v
                         i,j-1/2

        """

        # this returns u on x-interfaces and v on y-interfaces.  These
        # constitute the MAC grid
        if self.verbose > 0:
            print("  making MAC velocities")

        # create the coefficient to the grad (pi/beta) term
        coeff = self.aux_data.get_var("coeff")
        coeff.v()[:, :] = 1.0/rho.v()
        coeff.v()[:, :] = coeff.v()*beta0.v2d()
        self.aux_data.fill_BC("coeff")

        # create the source term
        source = self.aux_data.get_var("source_y")

        g = self.rp.get_param("lm-atmosphere.grav")
        rhoprime = self.make_prime(rho, rho0)
        source.v()[:, :] = rhoprime.v()*g/rho.v()
        self.aux_data.fill_BC("source_y")

        _um, _vm = lm_interface.mac_vels(myg.ng, myg.dx, myg.dy, self.dt,
                                           u, v,
                                           ldelta_ux, ldelta_vx,
                                           ldelta_uy, ldelta_vy,
                                           coeff*gradp_x, coeff*gradp_y,
                                           source)

        u_MAC = ai.ArrayIndexer(d=_um, grid=myg)
        v_MAC = ai.ArrayIndexer(d=_vm, grid=myg)

        # ---------------------------------------------------------------------
        # do a MAC projection to make the advective velocities divergence
        # free
        # ---------------------------------------------------------------------

        # we will solve D (beta_0^2/rho) G phi = D (beta_0 U^MAC), where
        # phi is cell centered, and U^MAC is the MAC-type staggered
        # grid of the advective velocities.

        if self.verbose > 0:
            print("  MAC projection")

        # create the coefficient array: beta0**2/rho
        # MZ!!!! probably don't need the buf here
        coeff.v(buf=1)[:, :] = 1.0/rho.v(buf=1)
        coeff.v(buf=1)[:, :] = coeff.v(buf=1)*beta0.v2d(buf=1)**2

        # create the multigrid object
        mg = vcMG.VarCoeffCCMG2d(myg.nx, myg.ny,
                                 xl_BC_type=self.cc_data.BCs["phi-MAC"].xlb,
                                 xr_BC_type=self.cc_data.BCs["phi-MAC"].xrb,
                                 yl_BC_type=self.cc_data.BCs["phi-MAC"].ylb,
                                 yr_BC_type=self.cc_data.BCs["phi-MAC"].yrb,
                                 xmin=myg.xmin, xmax=myg.xmax,
                                 ymin=myg.ymin, ymax=myg.ymax,
                                 coeffs=coeff,
                                 coeffs_bc=self.cc_data.BCs["density"],
                                 verbose=0)

        # first compute div{beta_0 U}
        div_beta_U = mg.soln_grid.scratch_array()

        # MAC velocities are edge-centered.  div{beta_0 U} is cell-centered.
        div_beta_U.v()[:, :] = \
            beta0.v2d()*(u_MAC.ip(1) - u_MAC.v())/myg.dx + \
            (beta0_edges.v2dp(1)*v_MAC.jp(1) -
             beta0_edges.v2d()*v_MAC.v())/myg.dy

        # solve the Poisson problem
        mg.init_RHS(div_beta_U)
        mg.solve(rtol=1.e-12)

        # update the normal velocities with the pressure gradient -- these
        # constitute our advective velocities.  Note that what we actually
        # solved for here is phi/beta_0
        phi_MAC = self.cc_data.get_var("phi-MAC")
        phi_MAC[:, :] = mg.get_solution(grid=myg)

        coeff = self.aux_data.get_var("coeff")
        coeff.v()[:, :] = 1.0/rho.v()
        coeff.v()[:, :] = coeff.v()*beta0.v2d()
        self.aux_data.fill_BC("coeff")

        coeff_x = myg.scratch_array()
        b = (3, 1, 0, 0)  # this seems more than we need
        coeff_x.v(buf=b)[:, :] = 0.5*(coeff.ip(-1, buf=b) + coeff.v(buf=b))

        coeff_y = myg.scratch_array()
        b = (0, 0, 3, 1)
        coeff_y.v(buf=b)[:, :] = 0.5*(coeff.jp(-1, buf=b) + coeff.v(buf=b))

        # we need the MAC velocities on all edges of the computational domain
        # here we do U = U - (beta_0/rho) grad (phi/beta_0)
        b = (0, 1, 0, 0)
        u_MAC.v(buf=b)[:, :] -= \
                coeff_x.v(buf=b)*(phi_MAC.v(buf=b) - phi_MAC.ip(-1, buf=b))/myg.dx

        b = (0, 0, 0, 1)
        v_MAC.v(buf=b)[:, :] -= \
                coeff_y.v(buf=b)*(phi_MAC.v(buf=b) - phi_MAC.jp(-1, buf=b))/myg.dy

        # ---------------------------------------------------------------------
        # predict rho to the edges and do its conservative update
        # ---------------------------------------------------------------------
        _rx, _ry = lm_interface.rho_states(myg.ng, myg.dx, myg.dy, self.dt,
                                             rho, u_MAC, v_MAC,
                                             ldelta_rx, ldelta_ry)

        rho_xint = ai.ArrayIndexer(d=_rx, grid=myg)
        rho_yint = ai.ArrayIndexer(d=_ry, grid=myg)

        rho_old = rho.copy()

        rho.v()[:, :] -= self.dt*(
            #  (rho u)_x
            (rho_xint.ip(1)*u_MAC.ip(1) - rho_xint.v()*u_MAC.v())/myg.dx +
            #  (rho v)_y
            (rho_yint.jp(1)*v_MAC.jp(1) - rho_yint.v()*v_MAC.v())/myg.dy)

        self.cc_data.fill_BC("density")

        # update eint as a diagnostic
        eint = self.cc_data.get_var("eint")
        gamma = self.rp.get_param("eos.gamma")
        eint.v()[:, :] = self.base["p0"].v2d()/(gamma - 1.0)/rho.v()

        # ---------------------------------------------------------------------
        # recompute the interface states, using the advective velocity
        # from above
        # ---------------------------------------------------------------------
        if self.verbose > 0:
            print("  making u, v edge states")

        coeff = self.aux_data.get_var("coeff")
        coeff.v()[:, :] = 2.0/(rho.v() + rho_old.v())
        coeff.v()[:, :] = coeff.v()*beta0.v2d()
        self.aux_data.fill_BC("coeff")

        _ux, _vx, _uy, _vy = \
               lm_interface.states(myg.ng, myg.dx, myg.dy, self.dt,
                                     u, v,
                                     ldelta_ux, ldelta_vx,
                                     ldelta_uy, ldelta_vy,
                                     coeff*gradp_x, coeff*gradp_y,
                                     source,
                                     u_MAC, v_MAC)

        u_xint = ai.ArrayIndexer(d=_ux, grid=myg)
        v_xint = ai.ArrayIndexer(d=_vx, grid=myg)
        u_yint = ai.ArrayIndexer(d=_uy, grid=myg)
        v_yint = ai.ArrayIndexer(d=_vy, grid=myg)

        # ---------------------------------------------------------------------
        # update U to get the provisional velocity field
        # ---------------------------------------------------------------------
        if self.verbose > 0:
            print("  doing provisional update of u, v")

        # compute (U.grad)U

        # we want u_MAC U_x + v_MAC U_y
        advect_x = myg.scratch_array()
        advect_y = myg.scratch_array()

        advect_x.v()[:, :] = \
            0.5*(u_MAC.v() + u_MAC.ip(1))*(u_xint.ip(1) - u_xint.v())/myg.dx +\
            0.5*(v_MAC.v() + v_MAC.jp(1))*(u_yint.jp(1) - u_yint.v())/myg.dy

        advect_y.v()[:, :] = \
            0.5*(u_MAC.v() + u_MAC.ip(1))*(v_xint.ip(1) - v_xint.v())/myg.dx +\
            0.5*(v_MAC.v() + v_MAC.jp(1))*(v_yint.jp(1) - v_yint.v())/myg.dy

        proj_type = self.rp.get_param("lm-atmosphere.proj_type")

        if proj_type == 1:
            u.v()[:, :] -= (self.dt*advect_x.v() + self.dt*gradp_x.v())
            v.v()[:, :] -= (self.dt*advect_y.v() + self.dt*gradp_y.v())

        elif proj_type == 2:
            u.v()[:, :] -= self.dt*advect_x.v()
            v.v()[:, :] -= self.dt*advect_y.v()

        # add the gravitational source
        rho_half = 0.5*(rho + rho_old)
        rhoprime = self.make_prime(rho_half, rho0)
        source[:, :] = (rhoprime*g/rho_half)
        self.aux_data.fill_BC("source_y")

        v[:, :] += self.dt*source

        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")

        if self.verbose > 0:
            print("min/max rho = {}, {}".format(self.cc_data.min("density"), self.cc_data.max("density")))
            print("min/max u   = {}, {}".format(self.cc_data.min("x-velocity"), self.cc_data.max("x-velocity")))
            print("min/max v   = {}, {}".format(self.cc_data.min("y-velocity"), self.cc_data.max("y-velocity")))

        # ---------------------------------------------------------------------
        # project the final velocity
        # ---------------------------------------------------------------------

        # now we solve L phi = D (U* /dt)
        if self.verbose > 0:
            print("  final projection")

        # create the coefficient array: beta0**2/rho
        coeff = 1.0/rho
        coeff.v()[:, :] = coeff.v()*beta0.v2d()**2

        # create the multigrid object
        mg = vcMG.VarCoeffCCMG2d(myg.nx, myg.ny,
                                 xl_BC_type=self.cc_data.BCs["phi"].xlb,
                                 xr_BC_type=self.cc_data.BCs["phi"].xrb,
                                 yl_BC_type=self.cc_data.BCs["phi"].ylb,
                                 yr_BC_type=self.cc_data.BCs["phi"].yrb,
                                 xmin=myg.xmin, xmax=myg.xmax,
                                 ymin=myg.ymin, ymax=myg.ymax,
                                 coeffs=coeff,
                                 coeffs_bc=self.cc_data.BCs["density"],
                                 verbose=0)

        # first compute div{beta_0 U}

        # u/v are cell-centered, divU is cell-centered
        div_beta_U.v()[:, :] = \
            0.5*beta0.v2d()*(u.ip(1) - u.ip(-1))/myg.dx + \
            0.5*(beta0.v2dp(1)*v.jp(1) - beta0.v2dp(-1)*v.jp(-1))/myg.dy

        mg.init_RHS(div_beta_U/self.dt)

        # use the old phi as our initial guess
        phiGuess = mg.soln_grid.scratch_array()
        phiGuess.v(buf=1)[:, :] = phi.v(buf=1)
        mg.init_solution(phiGuess)

        # solve
        mg.solve(rtol=1.e-12)

        # store the solution in our self.cc_data object -- include a single
        # ghostcell
        phi[:, :] = mg.get_solution(grid=myg)

        # get the cell-centered gradient of p and update the velocities
        # this differs depending on what we projected.
        gradphi_x, gradphi_y = mg.get_solution_gradient(grid=myg)

        # U = U - (beta_0/rho) grad (phi/beta_0)
        coeff = 1.0/rho
        coeff.v()[:, :] = coeff.v()*beta0.v2d()

        u.v()[:, :] -= self.dt*coeff.v()*gradphi_x.v()
        v.v()[:, :] -= self.dt*coeff.v()*gradphi_y.v()

        # store gradp for the next step

        if proj_type == 1:
            gradp_x.v()[:, :] += gradphi_x.v()
            gradp_y.v()[:, :] += gradphi_y.v()

        elif proj_type == 2:
            gradp_x.v()[:, :] = gradphi_x.v()
            gradp_y.v()[:, :] = gradphi_y.v()

        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")

        self.cc_data.fill_BC("gradp_x")
        self.cc_data.fill_BC("gradp_y")

        # increment the time
        if not self.in_preevolve:
            self.cc_data.t += self.dt
            self.n += 1

    def dovis(self):
        """
        Do runtime visualization
        """
        plt.clf()

        # plt.rc("font", size=10)

        rho = self.cc_data.get_var("density")
        rho0 = self.base["rho0"]
        rhoprime = self.make_prime(rho, rho0)

        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        myg = self.cc_data.grid

        magvel = np.sqrt(u**2 + v**2)

        vort = myg.scratch_array()

        dv = 0.5*(v.ip(1) - v.ip(-1))/myg.dx
        du = 0.5*(u.jp(1) - u.jp(-1))/myg.dy

        vort.v()[:, :] = dv - du

        fig, axes = plt.subplots(nrows=2, ncols=2, num=1)
        plt.subplots_adjust(hspace=0.25)

        fields = [rho, magvel, vort, rhoprime]
        field_names = [r"$\rho$", r"|U|", r"$\nabla \times U$", r"$\rho'$"]

        for n in range(len(fields)):
            ax = axes.flat[n]

            f = fields[n]

            img = ax.imshow(np.transpose(f.v()),
                            interpolation="nearest", origin="lower",
                            extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax], cmap=self.cm)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(field_names[n])

            plt.colorbar(img, ax=ax)

        plt.figtext(0.05, 0.0125, f"t = {self.cc_data.t:10.5f}")

        plt.pause(0.001)
        plt.draw()

    def write_extras(self, f):
        """
        Output simulation-specific data to the h5py file f
        """

        # we implement our own version to allow us to store the base
        # state

        gb = f.create_group("base state")

        for key in self.base:
            gb.create_dataset(key, data=self.base[key].d)

    def read_extras(self, f):
        """
        read in any simulation-specific data from an h5py file object f
        """

        gb = f["base state"]
        for name in gb:
            self.base[name] = Basestate(self.cc_data.grid.ny, ng=self.cc_data.grid.ng)
            self.base[name].d[:] = gb[name]
