from util import runparams
import mesh.reconstruction_f as reconstruction_f
import incomp_interface_f
import multigrid.multigrid as multigrid

def evolve(myData, dt):
    """ evolve the incompressible equations through one timestep """

    
    u = myData.getVarPtr("x-velocity")
    v = myData.getVarPtr("y-velocity")

    gradp_x = myData.getVarPtr("gradp_x")
    gradp_y = myData.getVarPtr("gradp_y")

    phi = myData.getVarPtr("phi")

    myg = myData.grid

    dtdx = dt/myg.dx
    dtdy = dt/myg.dy

    #-------------------------------------------------------------------------
    # create the limited slopes of u and v (in both directions)
    #-------------------------------------------------------------------------
    limiter = runparams.getParam("incompressible.limiter")
    if (limiter == 0):
        limitFunc = reconstruction_f.nolimit
    elif (limiter == 1):
        limitFunc = reconstruction_f.limit2
    else:
        limitFunc = reconstruction_f.limit4
    
    ldelta_ux = limitFunc(1, u, myg.qx, myg.qy, myg.ng)
    ldelta_vx = limitFunc(1, v, myg.qx, myg.qy, myg.ng)

    ldelta_uy = limitFunc(2, u, myg.qx, myg.qy, myg.ng)
    ldelta_vy = limitFunc(2, v, myg.qx, myg.qy, myg.ng)

    
    #-------------------------------------------------------------------------
    # get the advective velocities
    #-------------------------------------------------------------------------
    
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
    print "  making MAC velocities"

    u_MAC, v_MAC = incomp_interface_f.mac_vels(myg.qx, myg.qy, myg.ng, 
                                               myg.dx, myg.dy, dt,
                                               u, v,
                                               ldelta_ux, ldelta_vx,
                                               ldelta_uy, ldelta_vy,
                                               gradp_x, gradp_y)


    #-------------------------------------------------------------------------
    # do a MAC projection ot make the advective velocities divergence
    # free
    # -------------------------------------------------------------------------

    # we will solve L phi = D U^MAC, where phi is cell centered, and
    # U^MAC is the MAC-type staggered grid of the advective
    # velocities.

    print "  MAC projection"

    # create the multigrid object
    MG = multigrid.ccMG2d(myg.nx, myg.ny,
                          xlBCtype="periodic", xrBCtype="periodic",
                          ylBCtype="periodic", yrBCtype="periodic",
                          xmin=myg.xmin, xmax=myg.xmax,
                          ymin=myg.ymin, ymax=myg.ymax,
                          verbose=0)

    # first compute divU
    divU = MG.solnGrid.scratchArray()

    # MAC velocities are edge-centered.  divU is cell-centered.
    divU[MG.ilo:MG.ihi+1,MG.jlo:MG.jhi+1] = \
        (u_MAC[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] - 
         u_MAC[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx + \
        (v_MAC[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] - 
         v_MAC[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1])/myg.dy
    
    MG.initRHS(divU)
        
    # solve the Poisson problem
    MG.initZeros()
    MG.solve(rtol=1.e-12)

    # update the normal velocities with the pressure gradient -- these
    # constitute our advective velocities
    phi_MAC = myData.getVarPtr("phi-MAC")
    solution = MG.getSolution()

    phi_MAC[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2] = \
        solution[MG.ilo-1:MG.ihi+2,MG.jlo-1:MG.jhi+2]

    # we need the MAC velocities on all edges of the computational domain
    u_MAC[myg.ilo:myg.ihi+2,myg.jlo:myg.jhi+1] -= \
        (phi_MAC[myg.ilo  :myg.ihi+2,myg.jlo:myg.jhi+1] -
         phi_MAC[myg.ilo-1:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx

    v_MAC[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+2] -= \
        (phi_MAC[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+2] -
         phi_MAC[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi+1])/myg.dy


    #-------------------------------------------------------------------------
    # recompute the interface states, using the advective velocity
    # from above
    # -------------------------------------------------------------------------
    print "  making u, v edge states"

    u_xint, v_xint, u_yint, v_yint = \
        incomp_interface_f.states(myg.qx, myg.qy, myg.ng, 
                                  myg.dx, myg.dy, dt,
                                  u, v,
                                  ldelta_ux, ldelta_vx,
                                  ldelta_uy, ldelta_vy,
                                  gradp_x, gradp_y,
                                  u_MAC, v_MAC)


    #-------------------------------------------------------------------------
    # update U to get the provisional velocity field
    #-------------------------------------------------------------------------

    print "  doing provisional update of u, v"

    # compute (U.grad)U

    # we want u_MAC U_x + v_MAC U_y
    advect_x = myg.scratchArray()
    advect_y = myg.scratchArray()

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

             
    proj_type = runparams.getParam("incompressible.proj_type")

    if (proj_type == 1):
        u[:,:] -= (dt*advect_x[:,:] + dt*gradp_x[:,:])
        v[:,:] -= (dt*advect_y[:,:] + dt*gradp_y[:,:])

    elif (proj_type == 2):
        u[:,:] -= dt*advect_x[:,:]
        v[:,:] -= dt*advect_y[:,:]

    myData.fillBC("x-velocity")
    myData.fillBC("y-velocity")


    #-------------------------------------------------------------------------
    # project the final velocity
    #-------------------------------------------------------------------------

    # now we solve L phi = D (U* /dt)
    print "  final projection"
    
    # create the multigrid object
    MG = multigrid.ccMG2d(myg.nx, myg.ny,
                          xlBCtype="periodic", xrBCtype="periodic",
                          ylBCtype="periodic", yrBCtype="periodic",
                          xmin=myg.xmin, xmax=myg.xmax,
                          ymin=myg.ymin, ymax=myg.ymax,
                          verbose=0)

    # first compute divU

    # u/v are cell-centered, divU is cell-centered    
    divU[MG.ilo:MG.ihi+1,MG.jlo:MG.jhi+1] = \
        0.5*(u[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] - 
             u[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1])/myg.dx + \
        0.5*(v[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] - 
             v[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])/myg.dy
    
    MG.initRHS(divU/dt)

    # use the old phi as our initial guess
    phiGuess = MG.solnGrid.scratchArray()
    phiGuess[MG.ilo-1:MG.ihi+2,MG.jlo-1:MG.jhi+2] = \
        phi[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2]
    MG.initSolution(phiGuess)

    # solve
    MG.solve(rtol=1.e-12)

    # store the solution
    solution = MG.getSolution()

    phi[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2] = \
        solution[MG.ilo-1:MG.ihi+2,MG.jlo-1:MG.jhi+2]

    # compute the cell-centered gradient of p and update the velocities
    # this differs depending on what we projected.
    gradphi_x = myg.scratchArray()
    gradphi_y = myg.scratchArray()

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

    myData.fillBC("x-velocity")
    myData.fillBC("y-velocity")
