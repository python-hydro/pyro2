import mesh.reconstruction_f as reconstruction_f

def unsplitFluxes(my_data, rp, dt, scalar_name):
    """
    Construct the fluxes through the interfaces for the linear advection
    equation:

      a  + u a  + v a  = 0
       t      x      y

    We use a second-order (piecewise linear) unsplit Godunov method
    (following Colella 1990).

    In the pure advection case, there is no Riemann problem we need to
    solve -- we just simply do upwinding.  So there is only one 'state'
    at each interface, and the zone the information comes from depends
    on the sign of the velocity.

    Our convection is that the fluxes are going to be defined on the
    left edge of the computational zones


     |             |             |             |
     |             |             |             |
    -+------+------+------+------+------+------+--
     |     i-1     |      i      |     i+1     |

              a_l,i  a_r,i   a_l,i+1


    a_r,i and a_l,i+1 are computed using the information in
    zone i,j.

    Parameters
    ----------
    my_data : CellCenterData2d object
        The data object containing the grid and advective scalar that
        we are advecting.
    rp : RuntimeParameters object
        The runtime parameters for the simulation
    dt : float
        The timestep we are advancing through.
    scalar_name : str
        The name of the variable contained in my_data that we are
        advecting

    Returns
    -------
    out : ndarray, ndarray
        The fluxes on the x- and y-interfaces

    """

    my_grid = my_data.grid

    a = my_data.get_var(scalar_name)

    # get the advection velocities
    u = rp.get_param("advection.u")
    v = rp.get_param("advection.v")

    cx = u*dt/my_grid.dx
    cy = v*dt/my_grid.dy

    qx = my_grid.qx
    qy = my_grid.qy

    #--------------------------------------------------------------------------
    # monotonized central differences
    #--------------------------------------------------------------------------

    limiter = rp.get_param("advection.limiter")
    if limiter == 0:
        limitFunc = reconstruction_f.nolimit
    elif limiter == 1:
        limitFunc = reconstruction_f.limit2
    else:
        limitFunc = reconstruction_f.limit4

    ldelta_a = limitFunc(1, a.d, qx, qy, my_grid.ng)
    a_x = my_grid.scratch_array()

    # upwind
    if u < 0:
        # a_x[i,j] = a[i,j] - 0.5*(1.0 + cx)*ldelta_a[i,j]
        a_x[:,:] = a.d[:,:] - 0.5*(1.0 + cx)*ldelta_a[:,:]
    else:
        # a_x[i,j] = a[i-1,j] + 0.5*(1.0 - cx)*ldelta_a[i-1,j]
        a_x[1:,:] = a.d[0:qx-1,:] + 0.5*(1.0 - cx)*ldelta_a[0:qx-1,:]


    # y-direction
    ldelta_a = limitFunc(2, a.d, qx, qy, my_grid.ng)
    a_y = my_grid.scratch_array()

    # upwind
    if v < 0:
        # a_y[i,j] = a[i,j] - 0.5*(1.0 + cy)*ldelta_a[i,j]
        a_y[:,:] = a.d[:,:] - 0.5*(1.0 + cy)*ldelta_a[:,:]
    else:
        # a_y[i,j] = a[i,j-1] + 0.5*(1.0 - cy)*ldelta_a[i,j-1]
        a_y[:,1:] = a.d[:,0:qy-1] + 0.5*(1.0 - cy)*ldelta_a[:,0:qy-1]


    # compute the transverse flux differences.  The flux is just (u a)
    # HOTF
    F_xt = u*a_x
    F_yt = v*a_y

    F_x = my_grid.scratch_array()
    F_y = my_grid.scratch_array()

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


    dtdx2 = 0.5*dt/my_grid.dx
    dtdy2 = 0.5*dt/my_grid.dy

    for i in range(my_grid.ilo, my_grid.ihi+2):
        for j in range(my_grid.jlo, my_grid.jhi+2):
            F_x[i,j] = u*(a_x[i,j] - dtdy2*(F_yt[i+mx,j+1] - F_yt[i+mx,j]))

    for i in range(my_grid.ilo, my_grid.ihi+2):
        for j in range(my_grid.jlo, my_grid.jhi+2):
            F_y[i,j] = v*(a_y[i,j] - dtdx2*(F_xt[i+1,j+my] - F_xt[i,j+my]))

    return F_x, F_y
