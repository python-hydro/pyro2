import importlib

import matplotlib.pyplot as plt
import numpy as np

import pyro.incompressible as incompressible
import pyro.incompressible.incomp_interface as incomp_interface
import pyro.incompressible_viscous.BC as BC
import pyro.mesh.array_indexer as ai
import pyro.mesh.boundary as bnd
import pyro.mesh.patch as patch
import pyro.mesh.reconstruction as reconstruction
import pyro.multigrid.MG as MG
import pyro.particles.particles as particles
from pyro.simulation_null import bc_setup, grid_setup
from pyro.util import msg


class Simulation(incompressible.Simulation):

    def initialize(self):
        """
        Initialization of the data is the same as the incompressible
        solver, but we define user BC, and provide the viscosity
        as an auxiliary variable.
        """
        nu = self.rp.get_param("incompressible_viscous.viscosity")

        super().initialize(other_bc=True,
                           aux_vars=(("viscosity", nu),))

    def define_other_bc(self):
        bnd.define_bc("moving_lid", BC.user, is_solid=False)

    def evolve(self):
        """
        Solve is all the same steps as the incompressible solver, but the
        velocity update is different because of the viscosity term
        """
        super().evolve(other_update_velocity=True, other_source_term=True)

    def other_source_term(self):
        """
        This is the viscous term nu L U
        """
        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")
        myg = self.cc_data.grid
        nu = self.rp.get_param("incompressible_viscous.viscosity")

        source_x = myg.scratch_array()
        source_x.v()[:, :] = nu * ((u.ip(1) + u.ip(-1) - 2.0*u.v())/myg.dx**2 +
                                    (u.jp(1) + u.jp(-1) - 2.0*u.v())/myg.dy**2)

        source_y = myg.scratch_array()
        source_y.v()[:, :] = nu * ((v.ip(1) + v.ip(-1) - 2.0*v.v())/myg.dx**2 +
                                    (v.jp(1) + v.jp(-1) - 2.0*v.v())/myg.dy**2)

        return source_x, source_y

    def do_other_update_velocity(self, U_MAC, U_INT):
        """
        Solve for U in a (decoupled) parabolic solve including viscosity term
        """

        if self.verbose > 0:
            print("  doing parabolic solve for u, v")

        # Get variables
        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")
        gradp_x = self.cc_data.get_var("gradp_x")
        gradp_y = self.cc_data.get_var("gradp_y")
        myg = self.cc_data.grid
        nu = self.rp.get_param("incompressible_viscous.viscosity")

        # Get MAC and interface velocities from function args
        u_MAC, v_MAC = U_MAC
        u_xint, u_yint, v_xint, v_yint = U_INT

        # compute (U.grad)U terms

        # we want u_MAC U_x + v_MAC U_y
        advect_x = myg.scratch_array()
        advect_y = myg.scratch_array()

        # u u_x + v u_y
        advect_x.v()[:, :] = \
            0.5*(u_MAC.v() + u_MAC.ip(1))*(u_xint.ip(1) - u_xint.v())/myg.dx + \
            0.5*(v_MAC.v() + v_MAC.jp(1))*(u_yint.jp(1) - u_yint.v())/myg.dy

        # u v_x + v v_y
        advect_y.v()[:, :] = \
            0.5*(u_MAC.v() + u_MAC.ip(1))*(v_xint.ip(1) - v_xint.v())/myg.dx + \
            0.5*(v_MAC.v() + v_MAC.jp(1))*(v_yint.jp(1) - v_yint.v())/myg.dy

        # setup the MG object -- we want to solve a Helmholtz equation
        # equation of the form:
        # (alpha - beta L) U = f
        #
        # with alpha = 1
        #      beta  = (dt/2) nu
        #      f     = U + (dt/2) (nu L U - (U.grad)U - grad p)
        #
        # (one such equation for each velocity component)
        #
        # this is the form that arises with a Crank-Nicolson discretization
        # of the incompressible momentum equation

        # Solve for x-velocity

        mg = MG.CellCenterMG2d(myg.nx, myg.ny,
                        xmin=myg.xmin, xmax=myg.xmax,
                        ymin=myg.ymin, ymax=myg.ymax,
                        xl_BC_type=self.cc_data.BCs["x-velocity"].xlb,
                        xr_BC_type=self.cc_data.BCs["x-velocity"].xrb,
                        yl_BC_type=self.cc_data.BCs["x-velocity"].ylb,
                        yr_BC_type=self.cc_data.BCs["x-velocity"].yrb,
                        alpha=1.0, beta=0.5*self.dt*nu,
                        verbose=0)

        # form the RHS: f = u + (dt/2) nu L u  (where L is the Laplacian)
        f = mg.soln_grid.scratch_array()
        f.v()[:, :] = u.v() + 0.5*self.dt * nu * (
            (u.ip(1) + u.ip(-1) - 2.0*u.v())/myg.dx**2 +
            (u.jp(1) + u.jp(-1) - 2.0*u.v())/myg.dy**2)   # this is the diffusion part

        # f.v()[:, :] = u_MAC.v() + 0.5*self.dt * nu * (
        #     (u_MAC.ip(1) + u_MAC.ip(-1) - 2.0*u_MAC.v())/myg.dx**2 +
        #     (u_MAC.jp(1) + u_MAC.jp(-1) - 2.0*u_MAC.v())/myg.dy**2)   # this is the diffusion part

        f.v()[:, :] -= self.dt * (advect_x.v() + gradp_x.v())  # advection + pressure

        mg.init_RHS(f)

        # use the old u as our initial guess
        uGuess = mg.soln_grid.scratch_array()
        uGuess.v(buf=1)[:, :] = u.v(buf=1)
        mg.init_solution(uGuess)

        # solve
        mg.solve(rtol=1.e-12)

        # store the solution
        u.v()[:, :] = mg.get_solution().v()

        # Solve for y-velocity

        mg = MG.CellCenterMG2d(myg.nx, myg.ny,
                        xmin=myg.xmin, xmax=myg.xmax,
                        ymin=myg.ymin, ymax=myg.ymax,
                        xl_BC_type=self.cc_data.BCs["y-velocity"].xlb,
                        xr_BC_type=self.cc_data.BCs["y-velocity"].xrb,
                        yl_BC_type=self.cc_data.BCs["y-velocity"].ylb,
                        yr_BC_type=self.cc_data.BCs["y-velocity"].yrb,
                        alpha=1.0, beta=0.5*self.dt*nu,
                        verbose=0)

        # form the RHS: f = v + (dt/2) nu L v  (where L is the Laplacian)
        f = mg.soln_grid.scratch_array()
        f.v()[:, :] = v.v() + 0.5*self.dt * nu * (
            (v.ip(1) + v.ip(-1) - 2.0*v.v())/myg.dx**2 +
            (v.jp(1) + v.jp(-1) - 2.0*v.v())/myg.dy**2)

        # f.v()[:, :] = v_MAC.v() + 0.5*self.dt * nu * (
        #     (v_MAC.ip(1) + v_MAC.ip(-1) - 2.0*v_MAC.v())/myg.dx**2 +
        #     (v_MAC.jp(1) + v_MAC.jp(-1) - 2.0*v_MAC.v())/myg.dy**2)

        f.v()[:, :] -= self.dt * (advect_y.v() + gradp_y.v())

        mg.init_RHS(f)

        # use the old u as our initial guess
        uGuess = mg.soln_grid.scratch_array()
        uGuess.v(buf=1)[:, :] = v.v(buf=1)
        mg.init_solution(uGuess)

        # solve
        mg.solve(rtol=1.e-12)

        # store the solution
        v.v()[:, :] = mg.get_solution().v()

    def write_extras(self, f):
        """
        Output simulation-specific data to the h5py file f
        """

        # make note of the custom BC
        gb = f.create_group("BC")

        # the value here is the value of "is_solid"
        gb.create_dataset("moving_lid", data=False)
