from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from incompressible.problems import *
import incompressible.incomp_interface_f as incomp_interface_f
import mesh.reconstruction_f as reconstruction_f
import mesh.patch as patch
from simulation_null import NullSimulation, grid_setup, bc_setup
import multigrid.MG as MG
from util import profile

class Simulation(NullSimulation):

    def initialize(self):
        """
        Initialize the grid and variables for incompressible flow and
        set the initial conditions for the chosen problem.
        """

        my_grid = grid_setup(self.rp, ng=4)

        # create the variables
        bc, bc_xodd, bc_yodd = bc_setup(self.rp)
        
        my_data = patch.CellCenterData2d(my_grid)

        # velocities
        my_data.register_var("x-velocity", bc_xodd)
        my_data.register_var("y-velocity", bc_yodd)

        # phi -- used for the projections
        my_data.register_var("phi-MAC", bc)
        my_data.register_var("phi", bc)
        my_data.register_var("gradp_x", bc)
        my_data.register_var("gradp_y", bc)

        my_data.create()

        self.cc_data = my_data

        # now set the initial conditions for the problem
        exec(self.problem_name + '.init_data(self.cc_data, self.rp)')


    def compute_timestep(self):
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
        xtmp = self.cc_data.grid.dx/(abs(u))
        ytmp = self.cc_data.grid.dy/(abs(v))

        self.dt = cfl*min(xtmp.min(), ytmp.min())


    def preevolve(self):
        """
        preevolve is called before we being the timestepping loop.  For
        the incompressible solver, this does an initial projection on the
        velocity field and then goes through the full evolution to get the
        value of phi.  The fluid state (u, v) is then reset to values
        before this evolve.
        """

        self.in_preevolve = True

        myg = self.cc_data.grid

        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")


        # 1. do the initial projection.  This makes sure that our original
        # velocity field satisties div U = 0

        # next create the multigrid object.  We want Neumann BCs on phi
        # at solid walls and periodic on phi for periodic BCs
        mg = MG.CellCenterMG2d(myg.nx, myg.ny,
                               xl_BC_type="periodic",
                               xr_BC_type="periodic",
                               yl_BC_type="periodic",
                               yr_BC_type="periodic",
                               xmin=myg.xmin, xmax=myg.xmax,
                               ymin=myg.ymin, ymax=myg.ymax,
                               verbose=0)

        # first compute divU
        divU = mg.soln_grid.scratch_array()

        divU.v()[:,:] = \
            0.5*(u.ip(1) - u.ip(-1))/myg.dx + 0.5*(v.jp(1) - v.jp(-1))/myg.dy

        # solve L phi = DU

        # initialize our guess to the solution, set the RHS to divU and
        # solve
        mg.init_zeros()
        mg.init_RHS(divU.d)
        mg.solve(rtol=1.e-10)

        # store the solution in our self.cc_data object -- include a single
        # ghostcell
        phi = self.cc_data.get_var("phi")
        phi.d[:,:] = mg.get_solution(grid=myg).d


        # compute the cell-centered gradient of phi and update the
        # velocities
        gradp_x, gradp_y = mg.get_solution_gradient(grid=myg)

        u.d[:,:] -= gradp_x.d
        v.d[:,:] -= gradp_y.d

        # fill the ghostcells
        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")


        # 2. now get an approximation to gradp at n-1/2 by going through the
        # evolution.

        # store the current solution -- we'll restore it in a bit
        orig_data = patch.cell_center_data_clone(self.cc_data)

        # get the timestep
        self.compute_timestep()

        # evolve
        self.evolve()

        # update gradp_x and gradp_y in our main data object
        new_gp_x = self.cc_data.get_var("gradp_x")
        new_gp_y = self.cc_data.get_var("gradp_y")

        orig_gp_x = orig_data.get_var("gradp_x")
        orig_gp_y = orig_data.get_var("gradp_y")

        orig_gp_x.d[:,:] = new_gp_x.d[:,:]
        orig_gp_y.d[:,:] = new_gp_y.d[:,:]

        self.cc_data = orig_data

        if self.verbose > 0: print("done with the pre-evolution")

        self.in_preevolve = False


    def evolve(self):
        """
        Evolve the incompressible equations through one timestep.
        """

        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        gradp_x = self.cc_data.get_var("gradp_x")
        gradp_y = self.cc_data.get_var("gradp_y")

        phi = self.cc_data.get_var("phi")

        myg = self.cc_data.grid


        #---------------------------------------------------------------------
        # create the limited slopes of u and v (in both directions)
        #---------------------------------------------------------------------
        limiter = self.rp.get_param("incompressible.limiter")
        if (limiter == 0): limitFunc = reconstruction_f.nolimit
        elif (limiter == 1): limitFunc = reconstruction_f.limit2
        else: limitFunc = reconstruction_f.limit4

        ldelta_ux = limitFunc(1, u.d, myg.qx, myg.qy, myg.ng)
        ldelta_vx = limitFunc(1, v.d, myg.qx, myg.qy, myg.ng)

        ldelta_uy = limitFunc(2, u.d, myg.qx, myg.qy, myg.ng)
        ldelta_vy = limitFunc(2, v.d, myg.qx, myg.qy, myg.ng)

        #---------------------------------------------------------------------
        # get the advective velocities
        #---------------------------------------------------------------------

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
        if self.verbose > 0: print("  making MAC velocities")

        _um, _vm = incomp_interface_f.mac_vels(myg.qx, myg.qy, myg.ng,
                                               myg.dx, myg.dy, self.dt,
                                               u.d, v.d,
                                               ldelta_ux, ldelta_vx,
                                               ldelta_uy, ldelta_vy,
                                               gradp_x.d, gradp_y.d)

        u_MAC = patch.ArrayIndexer(d=_um, grid=myg)
        v_MAC = patch.ArrayIndexer(d=_vm, grid=myg)
        

        #---------------------------------------------------------------------
        # do a MAC projection ot make the advective velocities divergence
        # free
        #---------------------------------------------------------------------

        # we will solve L phi = D U^MAC, where phi is cell centered, and
        # U^MAC is the MAC-type staggered grid of the advective
        # velocities.

        if self.verbose > 0: print("  MAC projection")

        # create the multigrid object
        mg = MG.CellCenterMG2d(myg.nx, myg.ny,
                               xl_BC_type="periodic",
                               xr_BC_type="periodic",
                               yl_BC_type="periodic",
                               yr_BC_type="periodic",
                               xmin=myg.xmin, xmax=myg.xmax,
                               ymin=myg.ymin, ymax=myg.ymax,
                               verbose=0)

        # first compute divU
        divU = mg.soln_grid.scratch_array()

        # MAC velocities are edge-centered.  divU is cell-centered.
        divU.v()[:,:] = \
            (u_MAC.ip(1) - u_MAC.v())/myg.dx + (v_MAC.jp(1) - v_MAC.v())/myg.dy

        # solve the Poisson problem
        mg.init_zeros()
        mg.init_RHS(divU.d)
        mg.solve(rtol=1.e-12)

        # update the normal velocities with the pressure gradient -- these
        # constitute our advective velocities
        phi_MAC = self.cc_data.get_var("phi-MAC")
        solution = mg.get_solution()

        phi_MAC.v(buf=1)[:,:] = solution.v(buf=1)

        # we need the MAC velocities on all edges of the computational domain
        b = (0, 1, 0, 0)
        u_MAC.v(buf=b)[:,:] -= (phi_MAC.v(buf=b) - phi_MAC.ip(-1, buf=b))/myg.dx

        b = (0, 0, 0, 1)
        v_MAC.v(buf=b)[:,:] -= (phi_MAC.v(buf=b) - phi_MAC.jp(-1, buf=b))/myg.dy


        #---------------------------------------------------------------------
        # recompute the interface states, using the advective velocity
        # from above
        #---------------------------------------------------------------------
        if self.verbose > 0: print("  making u, v edge states")

        _ux, _vx, _uy, _vy = \
               incomp_interface_f.states(myg.qx, myg.qy, myg.ng,
                                         myg.dx, myg.dy, self.dt,
                                         u.d, v.d,
                                         ldelta_ux, ldelta_vx,
                                         ldelta_uy, ldelta_vy,
                                         gradp_x.d, gradp_y.d,
                                         u_MAC.d, v_MAC.d)

        u_xint = patch.ArrayIndexer(d=_ux, grid=myg)
        v_xint = patch.ArrayIndexer(d=_vx, grid=myg)
        u_yint = patch.ArrayIndexer(d=_uy, grid=myg)
        v_yint = patch.ArrayIndexer(d=_vy, grid=myg)

        #---------------------------------------------------------------------
        # update U to get the provisional velocity field
        #---------------------------------------------------------------------

        if self.verbose > 0: print("  doing provisional update of u, v")

        # compute (U.grad)U

        # we want u_MAC U_x + v_MAC U_y
        advect_x = myg.scratch_array()
        advect_y = myg.scratch_array()

        # u u_x + v u_y
        advect_x.v()[:,:] = \
            0.5*(u_MAC.v() + u_MAC.ip(1))*(u_xint.ip(1) - u_xint.v())/myg.dx + \
            0.5*(v_MAC.v() + v_MAC.jp(1))*(u_yint.jp(1) - u_yint.v())/myg.dy

        # u v_x + v v_y
        advect_y.v()[:,:] = \
            0.5*(u_MAC.v() + u_MAC.ip(1))*(v_xint.ip(1) - v_xint.v())/myg.dx + \
            0.5*(v_MAC.v() + v_MAC.jp(1))*(v_yint.jp(1) - v_yint.v())/myg.dy

        proj_type = self.rp.get_param("incompressible.proj_type")

        if proj_type == 1:
            u.d[:,:] -= (self.dt*advect_x.d[:,:] + self.dt*gradp_x.d[:,:])
            v.d[:,:] -= (self.dt*advect_y.d[:,:] + self.dt*gradp_y.d[:,:])

        elif proj_type == 2:
            u.d[:,:] -= self.dt*advect_x.d[:,:]
            v.d[:,:] -= self.dt*advect_y.d[:,:]

        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")


        #---------------------------------------------------------------------
        # project the final velocity
        #---------------------------------------------------------------------

        # now we solve L phi = D (U* /dt)
        if self.verbose > 0: print("  final projection")

        # create the multigrid object
        mg = MG.CellCenterMG2d(myg.nx, myg.ny,
                               xl_BC_type="periodic",
                               xr_BC_type="periodic",
                               yl_BC_type="periodic",
                               yr_BC_type="periodic",
                               xmin=myg.xmin, xmax=myg.xmax,
                               ymin=myg.ymin, ymax=myg.ymax,
                               verbose=0)

        # first compute divU

        # u/v are cell-centered, divU is cell-centered
        divU.v()[:,:] = \
            0.5*(u.ip(1) - u.ip(-1))/myg.dx + 0.5*(v.jp(1) - v.jp(-1))/myg.dy

        mg.init_RHS(divU.d/self.dt)

        # use the old phi as our initial guess
        phiGuess = mg.soln_grid.scratch_array()
        phiGuess.v(buf=1)[:,:] = phi.v(buf=1)
        mg.init_solution(phiGuess.d)

        # solve
        mg.solve(rtol=1.e-12)

        # store the solution
        phi.d[:,:] = mg.get_solution(grid=myg).d

        # compute the cell-centered gradient of p and update the velocities
        # this differs depending on what we projected.
        gradphi_x, gradphi_y = mg.get_solution_gradient(grid=myg)

        # u = u - grad_x phi dt
        u.d[:,:] -= self.dt*gradphi_x.d
        v.d[:,:] -= self.dt*gradphi_y.d

        # store gradp for the next step
        if proj_type == 1:
            gradp_x.d[:,:] += gradphi_x.d[:,:]
            gradp_y.d[:,:] += gradphi_y.d[:,:]

        elif proj_type == 2:
            gradp_x.d[:,:] = gradphi_x.d[:,:]
            gradp_y.d[:,:] = gradphi_y.d[:,:]

        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")

        # increment the time
        if not self.in_preevolve:
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

        vort = myg.scratch_array()
        divU = myg.scratch_array()

        vort.v()[:,:] = \
             0.5*(v.ip(1) - v.ip(-1))/myg.dx - \
             0.5*(u.jp(1) - u.jp(-1))/myg.dy

        divU.v()[:,:] = \
            0.5*(u.ip(1) - u.ip(-1))/myg.dx + \
            0.5*(v.jp(1) - v.jp(-1))/myg.dy

        fig, axes = plt.subplots(nrows=2, ncols=2, num=1)
        plt.subplots_adjust(hspace=0.25)

        fields = [u, v, vort, divU]
        field_names = ["u", "v", r"$\nabla \times U$", r"$\nabla \cdot U$"]

        for n in range(4):
            ax = axes.flat[n]

            f = fields[n]
            img = ax.imshow(np.transpose(f.v()),
                            interpolation="nearest", origin="lower",
                            extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(field_names[n])

            plt.colorbar(img, ax=ax)


        plt.figtext(0.05,0.0125, "t = %10.5f" % self.cc_data.t)

        plt.draw()
