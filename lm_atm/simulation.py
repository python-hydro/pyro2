from __future__ import print_function

import numpy as np
import pylab

from lm_atm.problems import *
import lm_atm.LM_atm_interface_f as lm_interface_f
import mesh.reconstruction_f as reconstruction_f
import mesh.patch as patch
import multigrid.variable_coeff_MG as vcMG
from util import profile

class Simulation:

    def __init__(self, problem_name, rp, timers=None):
        """
        Initialize the Simulation object for incompressible flow.

        Parameters
        ----------
        problem_name : str
            The name of the problem we wish to run.  This should
            correspond to one of the modules in LM-atmosphere/problems/
        rp : RuntimeParameters object
            The runtime parameters for the simulation
        timers : TimerCollection object, optional
            The timers used for profiling this simulation
        """

        self.rp = rp

        print(rp)

        self.cc_data = None
        self.aux_data = None
        self.base = {}

        self.problem_name = problem_name

        if timers == None:
            self.tc = profile.TimerCollection()
        else:
            self.tc = timers


    def initialize(self):
        """ 
        Initialize the grid and variables for low Mach atmospheric flow 
        and set the initial conditions for the chosen problem.
        """

        # setup the grid
        nx = self.rp.get_param("mesh.nx")
        ny = self.rp.get_param("mesh.ny")

        xmin = self.rp.get_param("mesh.xmin")
        xmax = self.rp.get_param("mesh.xmax")
        ymin = self.rp.get_param("mesh.ymin")
        ymax = self.rp.get_param("mesh.ymax")
    
        myg = patch.Grid2d(nx, ny, 
                           xmin=xmin, xmax=xmax, 
                           ymin=ymin, ymax=ymax, ng=4)

        # first figure out the BCs
        xlb_type = self.rp.get_param("mesh.xlboundary")
        xrb_type = self.rp.get_param("mesh.xrboundary")
        ylb_type = self.rp.get_param("mesh.ylboundary")
        yrb_type = self.rp.get_param("mesh.yrboundary")

        bc_dens = patch.BCObject(xlb=xlb_type, xrb=xrb_type, 
                                 ylb=ylb_type, yrb=yrb_type)

        # if we are reflecting, we need odd reflection in the normal
        # directions for the velocity
        bc_xodd = patch.BCObject(xlb=xlb_type, xrb=xrb_type, 
                                 ylb=ylb_type, yrb=yrb_type,
                                 odd_reflect_dir="x")

        bc_yodd = patch.BCObject(xlb=xlb_type, xrb=xrb_type, 
                                 ylb=ylb_type, yrb=yrb_type,
                                 odd_reflect_dir="y")
        
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
        for bc in [xlb_type, xrb_type, ylb_type, yrb_type]:
            if bc == "periodic":
                bctype = "periodic"
            elif bc in ["reflect", "slipwall"]:
                bctype = "neumann"
            elif bc in ["outflow"]:
                bctype = "dirichlet"
            bcs.append(bctype)

        bc_phi = patch.BCObject(xlb=bcs[0], xrb=bcs[1], ylb=bcs[2], yrb=bcs[3])

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
        self.base["rho0"] = np.zeros((myg.qy), dtype=np.float64)
        self.base["p0"] = np.zeros((myg.qy), dtype=np.float64)

        # now set the initial conditions for the problem 
        exec(self.problem_name + '.init_data(self.cc_data, self.base, self.rp)')

        # Construct beta_0
        gamma = self.rp.get_param("eos.gamma")
        self.base["beta0"] = self.base["p0"]**(1.0/gamma)

        # we'll also need beta_0 on vertical edges -- on the domain edges,
        # just do piecewise constant
        self.base["beta0-edges"] = np.zeros((myg.qy), dtype=np.float64)        
        self.base["beta0-edges"][myg.jlo+1:myg.jhi+1] = \
            0.5*(self.base["beta0"][myg.jlo  :myg.jhi] +
                 self.base["beta0"][myg.jlo+1:myg.jhi+1])
        self.base["beta0-edges"][myg.jlo] = self.base["beta0"][myg.jlo]
        self.base["beta0-edges"][myg.jhi+1] = self.base["beta0"][myg.jhi]
        

    def make_prime(self, a, a0):
        return a - a0[np.newaxis,:]


    def timestep(self):
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
        if not np.max(np.abs(u)) == 0:
            xtmp = np.min(myg.dx/(np.abs(u[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])))
        if not np.max(np.abs(v)) == 0:
            ytmp = np.min(myg.dy/(np.abs(v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])))

        dt = cfl*min(xtmp, ytmp)
                    
        # We need an alternate timestep that accounts for buoyancy, to
        # handle the case where the velocity is initially zero.
        rho = self.cc_data.get_var("density")    
        rho0 = self.base["rho0"]
        rhoprime = self.make_prime(rho, rho0)

        g = self.rp.get_param("lm-atmosphere.grav")

        F_buoy = np.max(np.abs(rhoprime[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]*g)/
                        rho[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])

        dt_buoy = np.sqrt(2.0*myg.dx/F_buoy)

        dt = min(dt, dt_buoy)
        print("timestep is {}".format(dt))

        return dt


    def preevolve(self):
        """ 
        preevolve is called before we being the timestepping loop.  For
        the low Mach solver, this does an initial projection on the
        velocity field and then goes through the full evolution to get the
        value of phi.  The fluid state (rho, u, v) is then reset to values
        before this evolve.
        """
        
        myg = self.cc_data.grid

        rho = self.cc_data.get_var("density")
        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")


        # 1. do the initial projection.  This makes sure that our original
        # velocity field satisties div U = 0

        # the coefficent for the elliptic equation is beta_0^2/rho
        coeff = 1.0/rho[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2]
        beta0 = self.base["beta0"]
        coeff = coeff*beta0[np.newaxis,myg.jlo-1:myg.jhi+2]**2

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
        div_beta_U[mg.ilo:mg.ihi+1,mg.jlo:mg.jhi+1] = \
            0.5*beta0[np.newaxis,mg.jlo:mg.jhi+1]* \
                (u[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] - 
                 u[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1])/myg.dx + \
            0.5*(beta0[np.newaxis,myg.jlo+1:myg.jhi+2]* \
                 v[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] - 
                 beta0[np.newaxis,myg.jlo-1:myg.jhi  ]*
                 v[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])/myg.dy

        # solve D (beta_0^2/rho) G (phi/beta_0) = D( beta_0 U )

        # set the RHS to divU and solve
        mg.init_RHS(div_beta_U)
        mg.solve(rtol=1.e-10)

        # store the solution in our self.cc_data object -- include a single
        # ghostcell
        phi = self.cc_data.get_var("phi")
        phi[:,:] = mg.get_solution(grid=myg)

        # get the cell-centered gradient of phi and update the 
        # velocities
        # FIXME: this update only needs to be done on the interior
        # cells -- not ghost cells
        gradp_x, gradp_y = mg.get_solution_gradient(grid=myg)

        coeff = 1.0/rho[:,:]
        coeff = coeff*beta0[np.newaxis,:]

        u[:,:] -= coeff*gradp_x
        v[:,:] -= coeff*gradp_y

        # fill the ghostcells
        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")


        # 2. now get an approximation to gradp at n-1/2 by going through the
        # evolution.

        # store the current solution -- we'll restore it in a bit
        orig_data = patch.cell_center_data_clone(self.cc_data)

        # get the timestep
        dt = self.timestep()

        # evolve
        self.evolve(dt)

        # update gradp_x and gradp_y in our main data object
        new_gp_x = self.cc_data.get_var("gradp_x")
        new_gp_y = self.cc_data.get_var("gradp_y")

        orig_gp_x = orig_data.get_var("gradp_x")
        orig_gp_y = orig_data.get_var("gradp_y")

        orig_gp_x[:,:] = new_gp_x[:,:]
        orig_gp_y[:,:] = new_gp_y[:,:]

        self.cc_data = orig_data

        print("done with the pre-evolution")


    def evolve(self, dt):
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

        dtdx = dt/myg.dx
        dtdy = dt/myg.dy

        #---------------------------------------------------------------------
        # create the limited slopes of rho, u and v (in both directions)
        #---------------------------------------------------------------------
        limiter = self.rp.get_param("lm-atmosphere.limiter")
        if limiter == 0: limitFunc = reconstruction_f.nolimit
        elif limiter == 1: limitFunc = reconstruction_f.limit2
        else: limitFunc = reconstruction_f.limit4
    

        ldelta_rx = limitFunc(1, rho, myg.qx, myg.qy, myg.ng)
        ldelta_ux = limitFunc(1, u, myg.qx, myg.qy, myg.ng)
        ldelta_vx = limitFunc(1, v, myg.qx, myg.qy, myg.ng)

        ldelta_ry = limitFunc(2, rho, myg.qx, myg.qy, myg.ng)
        ldelta_uy = limitFunc(2, u, myg.qx, myg.qy, myg.ng)
        ldelta_vy = limitFunc(2, v, myg.qx, myg.qy, myg.ng)
    
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
        print("  making MAC velocities")

        # create the coefficient to the grad (pi/beta) term
        coeff = self.aux_data.get_var("coeff")
        coeff[:,:] = 1.0/rho[:,:]
        coeff[:,:] = coeff*beta0[np.newaxis,:]
        self.aux_data.fill_BC("coeff")
        
        # create the source term
        source = self.aux_data.get_var("source_y")

        g = self.rp.get_param("lm-atmosphere.grav")        
        rhoprime = self.make_prime(rho, rho0)
        source[:,:] = rhoprime*g/rho
        self.aux_data.fill_BC("source_y")
        
        u_MAC, v_MAC = lm_interface_f.mac_vels(myg.qx, myg.qy, myg.ng, 
                                               myg.dx, myg.dy, dt,
                                               u, v,
                                               ldelta_ux, ldelta_vx,
                                               ldelta_uy, ldelta_vy,
                                               coeff*gradp_x, coeff*gradp_y, 
                                               source)


        #---------------------------------------------------------------------
        # do a MAC projection to make the advective velocities divergence
        # free
        #---------------------------------------------------------------------

        # we will solve D (beta_0^2/rho) G phi = D (beta_0 U^MAC), where
        # phi is cell centered, and U^MAC is the MAC-type staggered
        # grid of the advective velocities.  

        print("  MAC projection")

        # create the coefficient array: beta0**2/rho
        coeff = 1.0/rho[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2]
        coeff = coeff*beta0[np.newaxis,myg.jlo-1:myg.jhi+2]**2
        
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
        div_beta_U[mg.ilo:mg.ihi+1,mg.jlo:mg.jhi+1] = \
            beta0[np.newaxis,myg.jlo:myg.jhi+1]*(
                u_MAC[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] - 
                u_MAC[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx + \
            (beta0_edges[np.newaxis,myg.jlo+1:myg.jhi+2]*
             v_MAC[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] - 
             beta0_edges[np.newaxis,myg.jlo  :myg.jhi+1]*
             v_MAC[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1])/myg.dy
    
        # solve the Poisson problem
        mg.init_RHS(div_beta_U)
        mg.solve(rtol=1.e-12)


        # update the normal velocities with the pressure gradient -- these
        # constitute our advective velocities.  Note that what we actually
        # solved for here is phi/beta_0
        phi_MAC = self.cc_data.get_var("phi-MAC")
        phi_MAC[:,:] = mg.get_solution(grid=myg)

        coeff = self.aux_data.get_var("coeff")        
        coeff[:,:] = 1.0/rho[:,:]
        coeff[:,:] = coeff*beta0[np.newaxis,:]
        self.aux_data.fill_BC("coeff")
        
        coeff_x = myg.scratch_array()
        coeff_x[myg.ilo-3:myg.ihi+2,myg.jlo:myg.jhi+1] = \
                0.5*(coeff[myg.ilo-2:myg.ihi+3,myg.jlo:myg.jhi+1] +
                     coeff[myg.ilo-3:myg.ihi+2,myg.jlo:myg.jhi+1])
                       
        coeff_y = myg.scratch_array()
        coeff_y[myg.ilo:myg.ihi+1,myg.jlo-3:myg.jhi+2] = \
                0.5*(coeff[myg.ilo:myg.ihi+1,myg.jlo-2:myg.jhi+3] +
                     coeff[myg.ilo:myg.ihi+1,myg.jlo-3:myg.jhi+2])
                       
        # we need the MAC velocities on all edges of the computational domain
        # here we do U = U - (beta_0/rho) grad (phi/beta_0)
        u_MAC[myg.ilo:myg.ihi+2,myg.jlo:myg.jhi+1] -= \
                coeff_x[myg.ilo  :myg.ihi+2,myg.jlo:myg.jhi+1]* \
                (phi_MAC[myg.ilo  :myg.ihi+2,myg.jlo:myg.jhi+1] -
                 phi_MAC[myg.ilo-1:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx

        v_MAC[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+2] -= \
                coeff_y[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+2]* \
                (phi_MAC[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+2] -
                 phi_MAC[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi+1])/myg.dy


        #---------------------------------------------------------------------
        # predict rho to the edges and do its conservative update
        #---------------------------------------------------------------------
        rho_xint, rho_yint = lm_interface_f.rho_states(myg.qx, myg.qy, myg.ng, 
                                                       myg.dx, myg.dy, dt,
                                                       rho, u_MAC, v_MAC,
                                                       ldelta_rx, ldelta_ry)

        rho_old = rho.copy()

        rho[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] -= dt*(
            #  (rho u)_x
            (rho_xint[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1]*
             u_MAC[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
             rho_xint[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]*
             u_MAC[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx +
            #  (rho v)_y
            (rho_yint[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2]*
             v_MAC[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
             rho_yint[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]*
             v_MAC[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dy )
                        
        self.cc_data.fill_BC("density")

        #---------------------------------------------------------------------
        # recompute the interface states, using the advective velocity
        # from above
        #---------------------------------------------------------------------
        print("  making u, v edge states")

        coeff = self.aux_data.get_var("coeff")
        coeff[:,:] = 2.0/(rho[:,:] + rho_old[:,:])
        coeff[:,:] = coeff*beta0[np.newaxis,:]
        self.aux_data.fill_BC("coeff")
        
        u_xint, v_xint, u_yint, v_yint = \
               lm_interface_f.states(myg.qx, myg.qy, myg.ng, 
                                     myg.dx, myg.dy, dt,
                                     u, v,
                                     ldelta_ux, ldelta_vx,
                                     ldelta_uy, ldelta_vy,
                                     coeff*gradp_x, coeff*gradp_y,
                                     source, 
                                     u_MAC, v_MAC)


        #---------------------------------------------------------------------
        # update U to get the provisional velocity field
        #---------------------------------------------------------------------
        print("  doing provisional update of u, v")

        # compute (U.grad)U

        # we want u_MAC U_x + v_MAC U_y
        advect_x = myg.scratch_array()
        advect_y = myg.scratch_array()

        advect_x[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
            0.5*(u_MAC[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1] + 
                 u_MAC[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1]) * \
            (u_xint[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] - 
             u_xint[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx + \
            0.5*(v_MAC[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1] + 
                 v_MAC[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2]) * \
            (u_yint[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] - 
             u_yint[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1])/myg.dy 

        advect_y[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
            0.5*(u_MAC[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1] + 
                 u_MAC[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1]) * \
            (v_xint[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] - 
             v_xint[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx + \
            0.5*(v_MAC[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1] + 
                 v_MAC[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2]) * \
            (v_yint[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] - 
             v_yint[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1])/myg.dy 

             
        proj_type = self.rp.get_param("lm-atmosphere.proj_type")


        if proj_type == 1:
            u[:,:] -= (dt*advect_x[:,:] + dt*gradp_x[:,:])
            v[:,:] -= (dt*advect_y[:,:] + dt*gradp_y[:,:])

        elif proj_type == 2:
            u[:,:] -= dt*advect_x[:,:]
            v[:,:] -= dt*advect_y[:,:]


        # add the gravitational source
        rho_half = 0.5*(rho + rho_old)
        rhoprime = self.make_prime(rho_half, rho0)
        source = rhoprime*g/rho_half

        v[:,:] += dt*source
        
        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")

        print("min/max rho = {}, {}".format(np.min(rho), np.max(rho)))
        print("min/max u   = {}, {}".format(np.min(u), np.max(u)))
        print("min/max v   = {}, {}".format(np.min(v), np.max(v)))


        #---------------------------------------------------------------------
        # project the final velocity
        #---------------------------------------------------------------------

        # now we solve L phi = D (U* /dt)
        print("  final projection")

        # create the coefficient array: beta0**2/rho
        coeff = 1.0/rho[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2]
        coeff = coeff*beta0[np.newaxis,myg.jlo-1:myg.jhi+2]**2
    
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
        div_beta_U[mg.ilo:mg.ihi+1,mg.jlo:mg.jhi+1] = \
            0.5*beta0[np.newaxis,mg.jlo:mg.jhi+1]* \
                (u[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] - 
                 u[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1])/myg.dx + \
            0.5*(beta0[np.newaxis,myg.jlo+1:myg.jhi+2]* \
                 v[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] - 
                 beta0[np.newaxis,myg.jlo-1:myg.jhi  ]*
                 v[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])/myg.dy
    
        mg.init_RHS(div_beta_U/dt)

        # use the old phi as our initial guess
        phiGuess = mg.soln_grid.scratch_array()
        phiGuess[mg.ilo-1:mg.ihi+2,mg.jlo-1:mg.jhi+2] = \
           phi[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2]
        mg.init_solution(phiGuess)

        # solve
        mg.solve(rtol=1.e-12)


        # store the solution in our self.cc_data object -- include a single
        # ghostcell
        phi[:,:] = mg.get_solution(grid=myg)

        # get the cell-centered gradient of p and update the velocities
        # this differs depending on what we projected.
        gradphi_x, gradphi_y = mg.get_solution_gradient(grid=myg)

        
        # U = U - (beta_0/rho) grad (phi/beta_0)
        coeff = 1.0/rho[:,:]
        coeff = coeff*beta0[np.newaxis,:]

        u[:,:] -= dt*coeff*gradphi_x
        v[:,:] -= dt*coeff*gradphi_y
        
        # store gradp for the next step

        if proj_type == 1:
            gradp_x[:,:] += gradphi_x[:,:]
            gradp_y[:,:] += gradphi_y[:,:]

        elif proj_type == 2:
            gradp_x[:,:] = gradphi_x[:,:]
            gradp_y[:,:] = gradphi_y[:,:]
            
        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")

        self.cc_data.fill_BC("gradp_x")
        self.cc_data.fill_BC("gradp_y")        


    def dovis(self):
        """ 
        Do runtime visualization
        """
        pylab.clf()

        pylab.rc("font", size=10)

        rho = self.cc_data.get_var("density")

        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        #rho0 = self.base["rho0"]

        myg = self.cc_data.grid

        magvel = myg.scratch_array()
        vort = myg.scratch_array()

        magvel = np.sqrt(u**2 + v**2)

        #rhoprime = self.make_prime(rho, rho0)

        vort[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
             0.5*(v[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
                  v[myg.ilo-1:myg.ihi,myg.jlo:myg.jhi+1])/myg.dx - \
             0.5*(u[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
                  u[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi])/myg.dy

        fig, axes = pylab.subplots(nrows=2, ncols=2, num=1)
        pylab.subplots_adjust(hspace=0.25)

        fields = [rho, magvel, vort, rho]
        field_names = [r"$\rho$", r"|U|", r"$\nabla \times U$", r"$\rho$"]
    
        for n in range(4):
            ax = axes.flat[n]
    
            f = fields[n]
            img = ax.imshow(np.transpose(f[myg.ilo:myg.ihi+1,
                                           myg.jlo:myg.jhi+1]), 
                            interpolation="nearest", origin="lower",
                            extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(field_names[n])

            pylab.colorbar(img, ax=ax)


        pylab.figtext(0.05,0.0125, "t = %10.5f" % self.cc_data.t)

        pylab.draw()


    def finalize(self):
        """
        Do any final clean-ups for the simulation and call the problem's
        finalize() method.
        """
        exec(self.problem_name + '.finalize()')
