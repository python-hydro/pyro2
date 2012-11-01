from util import runparams


def evolve(myData, dt):
    """ evolve the incompressible equations through one timestep """

    
    u = myData.getVarPtr("x-velocity")
    v = myData.getVarPtr("x-velocity")

    gradp_x = myData.getVarPtr("gradp_y")
    gradp_y = myData.getVarPtr("gradp_x")

    myg = myData.grid

    dtdx = dt/myg.dx
    dtdy = dt/myg.dy

    #-------------------------------------------------------------------------
    # create the limited slopes of u and v
    #-------------------------------------------------------------------------


    #-------------------------------------------------------------------------
    # create the transverse velocities to be used in the advection
    #-------------------------------------------------------------------------
    
    utrans, vtrans = \
        incomp_interface_f.trans_vels(myg.qx, myg.qy, myg.ng, 
                                      myg.dx, myg.dy, dt,
                                      u, v,
                                      ldelta_u, ldelta_v)

    


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
    u_MAC, v_MAC = incomp_interface_f.mac_vels(myg.qx, myg.qy, myg.ng, 
                                               myg.dx, myg.dy, dt,
                                               u, v,
                                               ldelta_u, ldelta_v,
                                               gradp_x, gradp_y,
                                               utrans, vtrans)


    #-------------------------------------------------------------------------
    # do a MAC projection ot make the advective velocities divergence
    # free
    # -------------------------------------------------------------------------

    # we will solve L phi = D U^MAC, where phi is cell centered, and
    # U^MAC is the MAC-type staggered grid of the advective
    # velocities.

    # create the multigrid object


    # create the divU source term

    
    # solve the Poisson problem


    # update the normal velocities with the pressure gradient -- these
    # constitute our advective velocities



    #-------------------------------------------------------------------------
    # recompute the interface states, using the advective velocity
    # from above
    # -------------------------------------------------------------------------
    u_x, v_x, u_y, v_y = incomp_interface_f.states(myg.qx, myg.qy, myg.ng, 
                                                   myg.dx, myg.dy, dt,
                                                   u, v,
                                                   gradp_x, gradp_y,
                                                   utrans, vtrans,
                                                   uadv, vadv)


    #-------------------------------------------------------------------------
    # compute (U.grad)U
    #-------------------------------------------------------------------------



    #-------------------------------------------------------------------------
    # update U to get the provisional velocity field
    #-------------------------------------------------------------------------



    #-------------------------------------------------------------------------
    # project the final velocity
    #-------------------------------------------------------------------------





    [flux_x, flux_y] =  unsplitFluxes(myData.grid, dt, dens)

    """
    do the differencing for the fluxes now.  Here, we use slices so we
    avoid slow loops in python.  This is equivalent to:

    myPatch.data[i,j] = myPatch.data[i,j] + \
                            dtdx*(flux_x[i,j] - flux_x[i+1,j]) + \
                            dtdy*(flux_y[i,j] - flux_y[i,j+1])
    """

    qx = myData.grid.qx
    qy = myData.grid.qy
    dens[0:qx-1,0:qy-1] = dens[0:qx-1,0:qy-1] + \
        dtdx*(flux_x[0:qx-1,0:qy-1] - flux_x[1:qx,0:qy-1]) + \
        dtdy*(flux_y[0:qx-1,0:qy-1] - flux_y[0:qx-1,1:qy])
            

    
