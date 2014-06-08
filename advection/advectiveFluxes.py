"""
solve

   a  + u a  + v a  = 0
    t      x      y

"""

import numpy

import mesh.reconstruction_f as reconstruction_f
from util import runparams

def unsplitFluxes(myGrid, dt, a):

    # get the advection velocities
    u = runparams.getParam("advection.u")
    v = runparams.getParam("advection.v")

    cx = u*dt/myGrid.dx
    cy = v*dt/myGrid.dy
    

    #--------------------------------------------------------------------------
    # monotonized central differences
    #--------------------------------------------------------------------------

    limiter = runparams.getParam("advection.limiter")
    if (limiter == 0):
        limitFunc = reconstruction_f.nolimit
    elif (limiter == 1):
        limitFunc = reconstruction_f.limit2
    else:
        limitFunc = reconstruction_f.limit4

    ldelta_a = limitFunc(1, a, myGrid.qx, myGrid.qy, myGrid.ng)



    # in the pure advection case, there is no Riemann problem we need to
    # solve -- we just simply do upwinding.  So there is only one 'state'
    # at each interface, and the zone the information comes from depends
    # on the sign of the velocity -- this should be vectorized.
    a_x = numpy.zeros((myGrid.qx,myGrid.qy), dtype=numpy.float64)
    
    """
    
    the fluxes are going to be defined on the left edge of the
    computational zones

            
     |             |             |             |
     |             |             |             |
    -+------+------+------+------+------+------+--
     |     i-1     |      i      |     i+1     |
    
              a_l,i  a_r,i   a_l,i+1
            
    
    a_r,i and a_l,i+1 are computed using the information in
    zone i,j.
    
    """

    qx = myGrid.qx
    qy = myGrid.qy

    # upwind
    if (u < 0):
        # a_x[i,j] = a[i,j] - 0.5*(1.0 + cx)*ldelta_a[i,j]
        a_x[:,:] = a[:,:] - 0.5*(1.0 + cx)*ldelta_a[:,:]
    else:
        # a_x[i,j] = a[i-1,j] + 0.5*(1.0 - cx)*ldelta_a[i-1,j]        
        a_x[1:,:] = a[0:qx-1,:] + 0.5*(1.0 - cx)*ldelta_a[0:qx-1,:]


    # y-direction
    ldelta_a = limitFunc(2, a, myGrid.qx, myGrid.qy, myGrid.ng)

    a_y = numpy.zeros((myGrid.qx,myGrid.qy), dtype=numpy.float64)

    
    # upwind
    if (v < 0):
        # a_y[i,j] = a[i,j] - 0.5*(1.0 + cy)*ldelta_a[i,j]
        a_y[:,:] = a[:,:] - 0.5*(1.0 + cy)*ldelta_a[:,:]
    else:
        # a_y[i,j] = a[i,j-1] + 0.5*(1.0 - cy)*ldelta_a[i,j-1]
        a_y[:,1:] = a[:,0:qy-1] + 0.5*(1.0 - cy)*ldelta_a[:,0:qy-1]


    # compute the transverse flux differences.  The flux is just (u a)
    # HOTF
    F_xt = u*a_x
    F_yt = v*a_y

    F_x = numpy.zeros((myGrid.qx,myGrid.qy), dtype=numpy.float64)
    F_y = numpy.zeros((myGrid.qx,myGrid.qy), dtype=numpy.float64)

    # the zone where we grab the transverse flux derivative from
    # depends on the sign of the advective velocity
    
    if u <= 0:
        mx = 0
    else:
        mx = -1


    if v <= 0:
        my = 0
    else:
        my = -1


    dtdx2 = 0.5*dt/myGrid.dx
    dtdy2 = 0.5*dt/myGrid.dy
        
    i = myGrid.ilo
    while (i <= myGrid.ihi+1):

        j = myGrid.jlo
        while (j <= myGrid.jhi+1):

            F_x[i,j] = u*(a_x[i,j] - dtdy2*(F_yt[i+mx,j+1] - F_yt[i+mx,j]))
            
            j += 1
        i += 1

    i = myGrid.ilo
    while (i <= myGrid.ihi+1):

        j = myGrid.jlo
        while (j <= myGrid.jhi+1):

            F_y[i,j] = v*(a_y[i,j] - dtdx2*(F_xt[i+1,j+my] - F_xt[i,j+my]))
            
            j += 1
        i += 1
        
            
    return [F_x, F_y]


    
