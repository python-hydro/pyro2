"""
Implementation of the Colella 2nd order unsplit Godunov scheme.  This
is a 2-dimensional implementation only.  We assume that the grid is
uniform, but it is relatively straightforward to relax this
assumption.
                                                                               
There are several different options for this solver (they are all
discussed in the Colella paper).

  limiter          = 1 to use the 2nd order MC limiter 
                   = 2 to use the 4th order MC limiter

  use_flattening   = t to use the multidimensional flattening 
                     algorithm at shocks

  delta, z0, z1      these are the flattening parameters.  The default
                     are the values listed in Colella 1990.

   j+3/2--+---------+---------+---------+ 
          |         |         |         |
     j+1 _|         |         |         |
          |         |         |         |
          |         |         |         |
   j+1/2--+---------XXXXXXXXXXX---------+ 
          |         X         X         |
       j _|         X         X         |
          |         X         X         |
          |         X         X         | 
   j-1/2--+---------XXXXXXXXXXX---------+ 
          |         |         |         |
     j-1 _|         |         |         |
          |         |         |         |
          |         |         |         |
   j-3/2--+---------+---------+---------+ 
          |    |    |    |    |    |    | 
              i-1        i        i+1  
        i-3/2     i-1/2     i+1/2     i+3/2  

We wish to solve  

  U_t + F^x_x + F^y_y = H
       
we want U_{i+1/2}^{n+1/2} -- the interface values that are input to
the Riemann problem through the faces for each zone.

Taylor expanding yields                                                        
                  
   n+1/2                     dU           dU
  U          = U   + 0.5 dx  --  + 0.5 dt -- 
   i+1/2,j,L    i,j          dx           dt 
                                                                               
 
                             dU             dF^x   dF^y 
             = U   + 0.5 dx  --  - 0.5 dt ( ---- + ---- - H )  
                i,j          dx              dx     dy  


                              dU      dF^x            dF^y
             = U   + 0.5 ( dx -- - dt ---- ) - 0.5 dt ---- + 0.5 dt H 
                i,j           dx       dx              dy    
                                                                          

                                  dt       dU           dF^y    
             = U   + 0.5 dx ( 1 - -- A^x ) --  - 0.5 dt ---- + 0.5 dt H 
                i,j               dx       dx            dy   


                                dt       _            dF^y   
             = U   + 0.5  ( 1 - -- A^x ) DU  - 0.5 dt ---- + 0.5 dt H   
                i,j             dx                     dy      

                     +----------+-----------+  +----+----+   +---+---+  
                                |                   |            |   

                    this is the monotonized   this is the   source term  
                    central difference term   transverse    
                                              flux term 

There are two components, the central difference in the normal to the
interface, and the transverse flux difference.  This is done for the
left and right sides of all 4 interfaces in a zone, which are then
used as input to the Riemann problem, yielding the 1/2 time interface
values,

     n+1/2 
    U  
     i+1/2,j   

Then, the zone average values are updated in the usual finite-volume
way:     

    n+1    n     dt    x  n+1/2       x  n+1/2 
   U    = U    + -- { F (U       ) - F (U       ) } 
    i,j    i,j   dx       i-1/2,j        i+1/2,j   


                 dt    y  n+1/2       y  n+1/2 
               + -- { F (U       ) - F (U       ) }  
                 dy       i,j-1/2        i,j+1/2  

Updating U_{i,j}: 

  -- We want to find the state to the left and right (or top and
     bottom) of each interface, ex. U_{i+1/2,j,[lr]}^{n+1/2}, and use
     them to solve a Riemann problem across each of the four
     interfaces.
                                                                               
  -- U_{i+1/2,j,[lr]}^{n+1/2} is comprised of two parts, the
     computation of the monotonized central differences in the normal
     direction (eqs. 2.8, 2.10) and the computation of the transverse
     derivatives, which requires the solution of a Riemann problem in
     the transverse direction (eqs. 2.9, 2.14).
                                                                               
       -- the monotonized central difference part is computed using
          the primitive variables.
                                                                               
       -- We compute the central difference part in both directions
          before doing the transverse flux differencing, since for the
          high-order transverse flux implementation, we use these as
          the input to the transverse Riemann problem.
"""

import numpy
import vars
import eos
import mesh.reconstruction_f as reconstruction_f
from util import runparams
from util import profile
import interface_f

def unsplitFluxes(myData, dt):
    """
    unsplitFluxes returns the fluxes through the x and y interfaces by
    doing an unsplit reconstruction of the interface values and then
    solving the Riemann problem through all the interfaces at once
                                                                               
    currently we assume a gamma-law EOS 

    grav is the gravitational acceleration in the y-direction            
    """

    pf = profile.timer("unsplitFluxes")
    pf.begin()
    
    myg = myData.grid


    #=========================================================================
    # compute the primitive variables
    #=========================================================================
    # Q = (rho, u, v, p)

    dens = myData.getVarPtr("density")
    xmom = myData.getVarPtr("x-momentum")
    ymom = myData.getVarPtr("y-momentum")
    ener = myData.getVarPtr("energy")

    r = dens

    # get the velocities
    u = xmom/dens
    v = ymom/dens

    # get the pressure
    e = (ener - 0.5*(xmom**2 + ymom**2)/dens)/dens

    p = eos.pres(dens, e)

    smallp = 1.e-10
    p = p.clip(smallp)   # apply a floor to the pressure
    

    #=========================================================================
    # compute the flattening coefficients
    #=========================================================================

    # there is a single flattening coefficient (xi) for all directions
    xi_x = numpy.zeros((myg.qx, myg.qx), dtype=numpy.float64)
    xi_y = numpy.zeros((myg.qx, myg.qx), dtype=numpy.float64)
    xi   = numpy.zeros((myg.qx, myg.qx), dtype=numpy.float64)
    
    smallp = 1.e-10

    delta = runparams.getParam("compressible.delta")
    z0 = runparams.getParam("compressible.z0")
    z1 = runparams.getParam("compressible.z1")

    xi_x = reconstruction_f.flatten(1, p, u, myg.qx, myg.qy, myg.ng, smallp, delta, z0, z1)
    xi_y = reconstruction_f.flatten(2, p, v, myg.qx, myg.qy, myg.ng, smallp, delta, z0, z1)

    xi = reconstruction_f.flatten_multid(xi_x, xi_y, p, myg.qx, myg.qy, myg.ng)


    #=========================================================================
    # x-direction
    #=========================================================================

    # monotonized central differences in x-direction
    pfa = profile.timer("limiting")
    pfa.begin()
    
    ldelta_r = numpy.zeros((myg.qx, myg.qx), dtype=numpy.float64)
    ldelta_u = numpy.zeros((myg.qx, myg.qx), dtype=numpy.float64)
    ldelta_v = numpy.zeros((myg.qx, myg.qx), dtype=numpy.float64)
    ldelta_p = numpy.zeros((myg.qx, myg.qx), dtype=numpy.float64)

    ldelta_r = xi*reconstruction_f.limit4(1, r, myg.qx, myg.qy, myg.ng)
    ldelta_u = xi*reconstruction_f.limit4(1, u, myg.qx, myg.qy, myg.ng)
    ldelta_v = xi*reconstruction_f.limit4(1, v, myg.qx, myg.qy, myg.ng)
    ldelta_p = xi*reconstruction_f.limit4(1, p, myg.qx, myg.qy, myg.ng)
    
    pfa.end()
    
    # left and right primitive variable states
    pfb = profile.timer("interfaceStates")
    pfb.begin()

    # (V_l, V_r) = interfaceStates(1, myg, dt, 
    #                              r, u, v, p, 
    #                              ldelta_r, ldelta_u, ldelta_v, ldelta_p)
    gamma = runparams.getParam("eos.gamma")

    V_l = numpy.zeros((myg.qx, myg.qy, vars.nvar),  dtype=numpy.float64)
    V_r = numpy.zeros((myg.qx, myg.qy, vars.nvar),  dtype=numpy.float64)

    (V_l, V_r) = interface_f.states(1, myg.qx, myg.qy, myg.ng, myg.dx, dt,
                                    vars.nvar,
                                    gamma,
                                    r, u, v, p,
                                    ldelta_r, ldelta_u, ldelta_v, ldelta_p)                                    
    
    pfb.end()
                    

    # transform interface states back into conserved variables
    U_xl = numpy.zeros((myg.qx, myg.qy, myData.nvar),  dtype=numpy.float64)
    U_xr = numpy.zeros((myg.qx, myg.qy, myData.nvar),  dtype=numpy.float64)

    U_xl[:,:,vars.idens] = V_l[:,:,vars.irho]
    U_xl[:,:,vars.ixmom] = V_l[:,:,vars.irho]*V_l[:,:,vars.iu]
    U_xl[:,:,vars.iymom] = V_l[:,:,vars.irho]*V_l[:,:,vars.iv]
    U_xl[:,:,vars.iener] = eos.rhoe(V_l[:,:,vars.ip]) + \
        0.5*V_l[:,:,vars.irho]*(V_l[:,:,vars.iu]**2 + V_l[:,:,vars.iv]**2)

    U_xr[:,:,vars.idens] = V_r[:,:,vars.irho]
    U_xr[:,:,vars.ixmom] = V_r[:,:,vars.irho]*V_r[:,:,vars.iu]
    U_xr[:,:,vars.iymom] = V_r[:,:,vars.irho]*V_r[:,:,vars.iv]
    U_xr[:,:,vars.iener] = eos.rhoe(V_r[:,:,vars.ip]) + \
        0.5*V_r[:,:,vars.irho]*(V_r[:,:,vars.iu]**2 + V_r[:,:,vars.iv]**2)



    #=========================================================================
    # y-direction
    #=========================================================================

    # monotonized central differences in y-direction
    pfa.begin()

    ldelta_r = xi*reconstruction_f.limit4(2, r, myg.qx, myg.qy, myg.ng)
    ldelta_u = xi*reconstruction_f.limit4(2, u, myg.qx, myg.qy, myg.ng)
    ldelta_v = xi*reconstruction_f.limit4(2, v, myg.qx, myg.qy, myg.ng)
    ldelta_p = xi*reconstruction_f.limit4(2, p, myg.qx, myg.qy, myg.ng)

    pfa.end()
    
    # left and right primitive variable states
    pfb.begin()

    # (V_l, V_r) = interfaceStates(2, myg, dt, 
    #                              r, u, v, p, 
    #                              ldelta_r, ldelta_u, ldelta_v, ldelta_p)

    (V_l, V_r) = interface_f.states(2, myg.qx, myg.qy, myg.ng, myg.dy, dt,
                                    vars.nvar,
                                    gamma,
                                    r, u, v, p,
                                    ldelta_r, ldelta_u, ldelta_v, ldelta_p)                                    

    pfb.end()


    # transform interface states back into conserved variables
    U_yl = numpy.zeros((myg.qx, myg.qy, myData.nvar),  dtype=numpy.float64)
    U_yr = numpy.zeros((myg.qx, myg.qy, myData.nvar),  dtype=numpy.float64)

    U_yl[:,:,vars.idens] = V_l[:,:,vars.irho]
    U_yl[:,:,vars.ixmom] = V_l[:,:,vars.irho]*V_l[:,:,vars.iu]
    U_yl[:,:,vars.iymom] = V_l[:,:,vars.irho]*V_l[:,:,vars.iv]
    U_yl[:,:,vars.iener] = eos.rhoe(V_l[:,:,vars.ip]) + \
        0.5*V_l[:,:,vars.irho]*(V_l[:,:,vars.iu]**2 + V_l[:,:,vars.iv]**2)

    U_yr[:,:,vars.idens] = V_r[:,:,vars.irho]
    U_yr[:,:,vars.ixmom] = V_r[:,:,vars.irho]*V_r[:,:,vars.iu]
    U_yr[:,:,vars.iymom] = V_r[:,:,vars.irho]*V_r[:,:,vars.iv]
    U_yr[:,:,vars.iener] = eos.rhoe(V_r[:,:,vars.ip]) + \
        0.5*V_r[:,:,vars.irho]*(V_r[:,:,vars.iu]**2 + V_r[:,:,vars.iv]**2)


    #=========================================================================
    # apply source terms
    #=========================================================================
    

    #=========================================================================
    # compute transverse fluxes
    #=========================================================================
    pfc = profile.timer("riemann")
    pfc.begin()

    #F_x = riemann(1, myg, U_xl, U_xr)
    #F_y = riemann(2, myg, U_yl, U_yr)

    F_x = numpy.zeros((myg.qx, myg.qy, vars.nvar),  dtype=numpy.float64)
    F_y = numpy.zeros((myg.qx, myg.qy, vars.nvar),  dtype=numpy.float64)

    F_x = interface_f.riemann_cgf(1, myg.qx, myg.qy, myg.ng, 
                                  vars.nvar, vars.idens, vars.ixmom, vars.iymom, vars.iener, 
                                  gamma, U_xl, U_xr)

    F_y = interface_f.riemann_cgf(2, myg.qx, myg.qy, myg.ng, 
                                  vars.nvar, vars.idens, vars.ixmom, vars.iymom, vars.iener, 
                                  gamma, U_yl, U_yr)

    pfc.end()

    #=========================================================================
    # construct the interface values of U now
    #=========================================================================

    """
    finally, we can construct the state perpendicular to the interface
    by adding the central difference part to the trasverse flux
    difference.

    The states that we represent by indices i,j are shown below
    (1,2,3,4):
            

      j+3/2--+----------+----------+----------+ 
             |          |          |          | 
             |          |          |          | 
        j+1 -+          |          |          | 
             |          |          |          | 
             |          |          |          |    1: U_xl[i,j,:] = U  
      j+1/2--+----------XXXXXXXXXXXX----------+                      i-1/2,j,L
             |          X          X          | 
             |          X          X          |  
          j -+        1 X 2        X          |    2: U_xr[i,j,:] = U 
             |          X          X          |                      i-1/2,j,R
             |          X    4     X          | 
      j-1/2--+----------XXXXXXXXXXXX----------+  
             |          |    3     |          |    3: U_yl[i,j,:] = U 
             |          |          |          |                      i,j-1/2,L
        j-1 -+          |          |          |   
             |          |          |          |  
             |          |          |          |    4: U_yr[i,j,:] = U 
      j-3/2--+----------+----------+----------+                      i,j-1/2,R
             |    |     |    |     |    |     |  
                 i-1         i         i+1      
           i-3/2      i-1/2      i+1/2      i+3/2 


    remember that the fluxes are stored on the left edge, so 

    F_x[i,j,:] = F_x 
                    i-1/2, j   

    F_y[i,j,:] = F_y   
                    i, j-1/2   
                                       
    """

    pfd = profile.timer("transverse flux addition")
    pfd.begin()

    U_xl[myg.ilo-2:myg.ihi+2,myg.jlo-2:myg.jhi+2,:] += \
        - 0.5*dt/myg.dy * (F_y[myg.ilo-3:myg.ihi+1,myg.jlo-1:myg.jhi+3,:] - \
                           F_y[myg.ilo-3:myg.ihi+1,myg.jlo-2:myg.jhi+2,:])    

    U_xr[myg.ilo-2:myg.ihi+2,myg.jlo-2:myg.jhi+2,:] += \
        - 0.5*dt/myg.dy * (F_y[myg.ilo-2:myg.ihi+2,myg.jlo-1:myg.jhi+3,:] - \
                           F_y[myg.ilo-2:myg.ihi+2,myg.jlo-2:myg.jhi+2,:])    

    U_yl[myg.ilo-2:myg.ihi+2,myg.jlo-2:myg.jhi+2,:] += \
        - 0.5*dt/myg.dx * (F_x[myg.ilo-1:myg.ihi+3,myg.jlo-3:myg.jhi+1,:] - \
                           F_x[myg.ilo-2:myg.ihi+2,myg.jlo-3:myg.jhi+1,:])
                
    U_yr[myg.ilo-2:myg.ihi+2,myg.jlo-2:myg.jhi+2,:] += \
        - 0.5*dt/myg.dx * (F_x[myg.ilo-1:myg.ihi+3,myg.jlo-2:myg.jhi+2,:] - \
                           F_x[myg.ilo-2:myg.ihi+2,myg.jlo-2:myg.jhi+2,:])

    # should vectorize this
    # j = myg.jlo-2
    # while (j <= myg.jhi+2):

    #     i = myg.ilo-2
    #     while (i <= myg.ihi+2):

    #         U_xl[i,j,:] = U_xl[i,j,:] - \
    #             0.5*dt/myg.dy * (F_y[i-1,j+1,:] - F_y[i-1,j,:])
                
    #         U_xr[i,j,:] = U_xr[i,j,:] - \
    #             0.5*dt/myg.dy * (F_y[i,j+1,:] - F_y[i,j,:])

    #         U_yl[i,j,:] = U_yl[i,j,:] - \
    #             0.5*dt/myg.dx * (F_x[i+1,j-1,:] - F_x[i,j-1,:])
                
    #         U_yr[i,j,:] = U_yr[i,j,:] - \
    #             0.5*dt/myg.dx * (F_x[i+1,j,:] - F_x[i,j,:])

    #         i += 1
    #     j += 1

    pfd.end()

    #=========================================================================
    # construct the fluxes normal to the interfaces
    #=========================================================================
    
    # up until now, F_x and F_y stored the transverse fluxes, now we
    # overwrite with the fluxes normal to the interfaces

    pfc.begin()
        
    #F_x = riemann(1, myg, U_xl, U_xr)
    #F_y = riemann(2, myg, U_yl, U_yr)

    F_x = interface_f.riemann_cgf(1, myg.qx, myg.qy, myg.ng, 
                                  vars.nvar, vars.idens, vars.ixmom, vars.iymom, vars.iener, 
                                  gamma, U_xl, U_xr)

    F_y = interface_f.riemann_cgf(2, myg.qx, myg.qy, myg.ng, 
                                  vars.nvar, vars.idens, vars.ixmom, vars.iymom, vars.iener, 
                                  gamma, U_yl, U_yr)

    pfc.end()

    #=========================================================================
    # apply artifical viscosity
    #=========================================================================

    pf.end()

    return F_x, F_y



