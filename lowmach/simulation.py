from __future__ import print_function

import numpy
import pylab

from incompressible.problems import *
import incompressible.incomp_interface_f as incomp_interface_f
import mesh.reconstruction_f as reconstruction_f
import mesh.patch as patch
import multigrid.MG as MG
from util import profile

class Simulation:

    def __init__(self, problem_name, rp, timers=None):
        """
        Initialize the Simulation object for incompressible flow.

        Parameters
        ----------
        problem_name : str
            The name of the problem we wish to run.  This should
            correspond to one of the modules in incompressible/problems/
        rp : RuntimeParameters object
            The runtime parameters for the simulation
        timers : TimerCollection object, optional
            The timers used for profiling this simulation
        """

        self.rp = rp
        self.cc_data = None

        self.problem_name = problem_name

        if timers == None:
            self.tc = profile.TimerCollection()
        else:
            self.tc = timers


    def initialize(self):
        """ 
        Initialize the grid and variables for incompressible flow and
        set the initial conditions for the chosen problem.
        """

        # setup the grid
        nx = self.rp.get_param("mesh.nx")
        ny = self.rp.get_param("mesh.ny")

        xmin = self.rp.get_param("mesh.xmin")
        xmax = self.rp.get_param("mesh.xmax")
        ymin = self.rp.get_param("mesh.ymin")
        ymax = self.rp.get_param("mesh.ymax")
    
        my_grid = patch.Grid2d(nx, ny, 
                               xmin=xmin, xmax=xmax, 
                               ymin=ymin, ymax=ymax, ng=4)


        # create the variables

        # first figure out the BCs
        xlb_type = self.rp.get_param("mesh.xlboundary")
        xrb_type = self.rp.get_param("mesh.xrboundary")
        ylb_type = self.rp.get_param("mesh.ylboundary")
        yrb_type = self.rp.get_param("mesh.yrboundary")

        bc = patch.BCObject(xlb=xlb_type, xrb=xrb_type, 
                            ylb=ylb_type, yrb=yrb_type)

        # if we are reflecting, we need odd reflection in the normal
        # directions for the velocity
        bc_xodd = patch.BCObject(xlb=xlb_type, xrb=xrb_type, 
                                 ylb=ylb_type, yrb=yrb_type,
                                 odd_reflect_dir="x")

        bc_yodd = patch.BCObject(xlb=xlb_type, xrb=xrb_type, 
                                 ylb=ylb_type, yrb=yrb_type,
                                 odd_reflect_dir="y")
        
        my_data = patch.CellCenterData2d(my_grid)

        # velocities
        my_data.register_var("x-velocity", bc_xodd)
        my_data.register_var("y-velocity", bc_yodd)

        # density
        my_data.register_var("density", bc)

        # phi -- used for the projections
        my_data.register_var("phi-MAC", bc)
        my_data.register_var("phi", bc)


        # gradp -- used in the projection and interface states.  The BCs here
        # are tricky.  If we are periodic, then it is periodic.  Otherwise,
        # we just want to do first-order extrapolation (homogeneous Neumann)
        my_data.register_var("gradp_x", bc)
        my_data.register_var("gradp_y", bc)

        my_data.create()

        self.cc_data = my_data

        # now set the initial conditions for the problem 
        exec(self.problem_name + '.init_data(self.cc_data, self.rp)')


    def timestep(self):
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

        dt = cfl*min(xtmp.min(), ytmp.min())

        return dt


    def preevolve(self):
        """ 
        preevolve is called before we being the timestepping loop.  For
        the incompressible solver, this does an initial projection on the
        velocity field and then goes through the full evolution to get the
        value of phi.  The fluid state (u, v) is then reset to values
        before this evolve.
        """
        
        myg = self.cc_data.grid

        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")


        # 1. do the initial projection.  This makes sure that our original
        # velocity field satisties div U = 0

        # next create the multigrid object.  We defined phi with
        # the right BCs previously
        mg = MG.CellCenterMG2d(myg.nx, myg.ny,
                               xl_BC_type=self.cc_data.BCs["phi"].xlb, 
                               xr_BC_type=self.cc_data.BCs["phi"].xrb,
                               yl_BC_type=self.cc_data.BCs["phi"].ylb, 
                               yr_BC_type=self.cc_data.BCs["phi"].yrb,
                               xmin=myg.xmin, xmax=myg.xmax,
                               ymin=myg.ymin, ymax=myg.ymax,
                               verbose=0)

        # first compute divU
        divU = mg.soln_grid.scratch_array()

        divU[mg.ilo:mg.ihi+1,mg.jlo:mg.jhi+1] = \
            0.5*(u[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] - 
                 u[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1])/myg.dx + \
            0.5*(v[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] - 
                 v[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])/myg.dy

        # solve L phi = DU

        # initialize our guess to the solution, set the RHS to divU and
        # solve
        mg.init_zeros()
        mg.init_RHS(divU)
        mg.solve(rtol=1.e-10)

        # store the solution in our self.cc_data object -- include a single
        # ghostcell
        phi = self.cc_data.get_var("phi")
        solution = mg.get_solution()

        phi[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2] = \
            solution[mg.ilo-1:mg.ihi+2,mg.jlo-1:mg.jhi+2]

        # compute the cell-centered gradient of phi and update the 
        # velocities
        gradp_x = myg.scratch_array()
        gradp_y = myg.scratch_array()

        gradp_x[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
            0.5*(phi[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
                 phi[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1])/myg.dx

        gradp_y[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
            0.5*(phi[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
                 phi[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])/myg.dy

        u[:,:] -= gradp_x
        v[:,:] -= gradp_y

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
        Evolve the incompressible equations through one timestep. 
        """
    
        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        gradp_x = self.cc_data.get_var("gradp_x")
        gradp_y = self.cc_data.get_var("gradp_y")

        phi = self.cc_data.get_var("phi")

        myg = self.cc_data.grid

        dtdx = dt/myg.dx
        dtdy = dt/myg.dy

        #---------------------------------------------------------------------
        # create the limited slopes of u and v (in both directions)
        #---------------------------------------------------------------------
        limiter = self.rp.get_param("incompressible.limiter")
        if (limiter == 0): limitFunc = reconstruction_f.nolimit
        elif (limiter == 1): limitFunc = reconstruction_f.limit2
        else: limitFunc = reconstruction_f.limit4
    
        ldelta_ux = limitFunc(1, u, myg.qx, myg.qy, myg.ng)
        ldelta_vx = limitFunc(1, v, myg.qx, myg.qy, myg.ng)

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

        u_MAC, v_MAC = incomp_interface_f.mac_vels(myg.qx, myg.qy, myg.ng, 
                                                   myg.dx, myg.dy, dt,
                                                   u, v,
                                                   ldelta_ux, ldelta_vx,
                                                   ldelta_uy, ldelta_vy,
                                                   gradp_x, gradp_y)


        #---------------------------------------------------------------------
        # do a MAC projection ot make the advective velocities divergence
        # free
        #---------------------------------------------------------------------

        # we will solve L phi = D U^MAC, where phi is cell centered, and
        # U^MAC is the MAC-type staggered grid of the advective
        # velocities.

        print("  MAC projection")

        # create the multigrid object
        mg = MG.CellCenterMG2d(myg.nx, myg.ny,
                               xl_BC_type=self.cc_data.BCs["phi-MAC"].xlb, 
                               xr_BC_type=self.cc_data.BCs["phi-MAC"].xrb,
                               yl_BC_type=self.cc_data.BCs["phi-MAC"].ylb, 
                               yr_BC_type=self.cc_data.BCs["phi-MAC"].yrb,
                               xmin=myg.xmin, xmax=myg.xmax,
                               ymin=myg.ymin, ymax=myg.ymax,
                               verbose=0)

        # first compute divU
        divU = mg.soln_grid.scratch_array()

        # MAC velocities are edge-centered.  divU is cell-centered.
        divU[mg.ilo:mg.ihi+1,mg.jlo:mg.jhi+1] = \
            (u_MAC[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] - 
             u_MAC[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx + \
            (v_MAC[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] - 
             v_MAC[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1])/myg.dy
    
        # solve the Poisson problem
        mg.init_zeros()
        mg.init_RHS(divU)
        mg.solve(rtol=1.e-12)

        # update the normal velocities with the pressure gradient -- these
        # constitute our advective velocities
        phi_MAC = self.cc_data.get_var("phi-MAC")
        solution = mg.get_solution()

        phi_MAC[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2] = \
            solution[mg.ilo-1:mg.ihi+2,mg.jlo-1:mg.jhi+2]

        # we need the MAC velocities on all edges of the computational domain
        u_MAC[myg.ilo:myg.ihi+2,myg.jlo:myg.jhi+1] -= \
            (phi_MAC[myg.ilo  :myg.ihi+2,myg.jlo:myg.jhi+1] -
             phi_MAC[myg.ilo-1:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx

        v_MAC[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+2] -= \
            (phi_MAC[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+2] -
             phi_MAC[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi+1])/myg.dy


        #---------------------------------------------------------------------
        # recompute the interface states, using the advective velocity
        # from above
        #---------------------------------------------------------------------
        print("  making u, v edge states")

        u_xint, v_xint, u_yint, v_yint = \
               incomp_interface_f.states(myg.qx, myg.qy, myg.ng, 
                                         myg.dx, myg.dy, dt,
                                         u, v,
                                         ldelta_ux, ldelta_vx,
                                         ldelta_uy, ldelta_vy,
                                         gradp_x, gradp_y,
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

             
        proj_type = self.rp.get_param("incompressible.proj_type")

        if (proj_type == 1):
            u[:,:] -= (dt*advect_x[:,:] + dt*gradp_x[:,:])
            v[:,:] -= (dt*advect_y[:,:] + dt*gradp_y[:,:])

        elif (proj_type == 2):
            u[:,:] -= dt*advect_x[:,:]
            v[:,:] -= dt*advect_y[:,:]

        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")


        #---------------------------------------------------------------------
        # project the final velocity
        #---------------------------------------------------------------------

        # now we solve L phi = D (U* /dt)
        print("  final projection")
    
        # create the multigrid object
        mg = MG.CellCenterMG2d(myg.nx, myg.ny,
                               xl_BC_type=self.cc_data.BCs["phi"].xlb, 
                               xr_BC_type=self.cc_data.BCs["phi"].xrb,
                               yl_BC_type=self.cc_data.BCs["phi"].ylb, 
                               yr_BC_type=self.cc_data.BCs["phi"].yrb,
                               xmin=myg.xmin, xmax=myg.xmax,
                               ymin=myg.ymin, ymax=myg.ymax,
                               verbose=0)

        # first compute divU

        # u/v are cell-centered, divU is cell-centered    
        divU[mg.ilo:mg.ihi+1,mg.jlo:mg.jhi+1] = \
            0.5*(u[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] - 
                 u[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1])/myg.dx + \
            0.5*(v[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] - 
                 v[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])/myg.dy
    
        mg.init_RHS(divU/dt)

        # use the old phi as our initial guess
        phiGuess = mg.soln_grid.scratch_array()
        phiGuess[mg.ilo-1:mg.ihi+2,mg.jlo-1:mg.jhi+2] = \
           phi[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2]
        mg.init_solution(phiGuess)

        # solve
        mg.solve(rtol=1.e-12)

        # store the solution
        solution = mg.get_solution()

        phi[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2] = \
            solution[mg.ilo-1:mg.ihi+2,mg.jlo-1:mg.jhi+2]

        # compute the cell-centered gradient of p and update the velocities
        # this differs depending on what we projected.
        gradphi_x = myg.scratch_array()
        gradphi_y = myg.scratch_array()

        gradphi_x[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
            0.5*(phi[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
                 phi[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1])/myg.dx

        gradphi_y[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
            0.5*(phi[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
                 phi[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])/myg.dy

        # u = u - grad_x phi dt
        u[:,:] -= dt*gradphi_x
        v[:,:] -= dt*gradphi_y
        
        # store gradp for the next step
        if (proj_type == 1):
            gradp_x[:,:] += gradphi_x[:,:]
            gradp_y[:,:] += gradphi_y[:,:]

        elif (proj_type == 2):
            gradp_x[:,:] = gradphi_x[:,:]
            gradp_y[:,:] = gradphi_y[:,:]
            
        self.cc_data.fill_BC("x-velocity")
        self.cc_data.fill_BC("y-velocity")


    def dovis(self):
        """ 
        Do runtime visualization
        """
        pylab.clf()

        pylab.rc("font", size=10)

        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        myg = self.cc_data.grid

        vort = myg.scratch_array()
        divU = myg.scratch_array()

        vort[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
             0.5*(v[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
                  v[myg.ilo-1:myg.ihi,myg.jlo:myg.jhi+1])/myg.dx - \
             0.5*(u[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
                  u[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi])/myg.dy

        divU[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
            0.5*(u[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
                 u[myg.ilo-1:myg.ihi,myg.jlo:myg.jhi+1])/myg.dx + \
            0.5*(v[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
                 v[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi])/myg.dy

        fig, axes = pylab.subplots(nrows=2, ncols=2, num=1)
        pylab.subplots_adjust(hspace=0.25)

        fields = [u, v, vort, divU]
        field_names = ["u", "v", r"$\nabla \times U$", r"$\nabla \cdot U$"]
    
        for n in range(4):
            ax = axes.flat[n]
    
            f = fields[n]
            img = ax.imshow(numpy.transpose(f[myg.ilo:myg.ihi+1,
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
