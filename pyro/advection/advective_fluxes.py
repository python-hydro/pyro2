import pyro.mesh.reconstruction as reconstruction


def unsplit_fluxes(my_data, rp, dt, scalar_name):
    r"""
    Construct the fluxes through the interfaces for the linear advection
    equation:

    .. math::

       a_t  + u a_x  + v a_y  = 0

    We use a second-order (piecewise linear) unsplit Godunov method
    (following Colella 1990).

    In the pure advection case, there is no Riemann problem we need to
    solve -- we just simply do upwinding.  So there is only one 'state'
    at each interface, and the zone the information comes from depends
    on the sign of the velocity.

    Our convection is that the fluxes are going to be defined on the
    left edge of the computational zones::

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

    myg = my_data.grid

    a = my_data.get_var(scalar_name)

    # get the advection velocities
    u = rp.get_param("advection.u")
    v = rp.get_param("advection.v")

    cx = u*dt/myg.dx
    cy = v*dt/myg.dy

    # --------------------------------------------------------------------------
    # monotonized central differences
    # --------------------------------------------------------------------------

    limiter = rp.get_param("advection.limiter")

    ldelta_ax = reconstruction.limit(a, myg, 1, limiter)
    ldelta_ay = reconstruction.limit(a, myg, 2, limiter)

    a_x = myg.scratch_array()

    # upwind
    if u < 0:
        # a_x[i,j] = a[i,j] - 0.5*(1.0 + cx)*ldelta_a[i,j]
        a_x.v(buf=1)[:, :] = a.v(buf=1) - 0.5*(1.0 + cx)*ldelta_ax.v(buf=1)
    else:
        # a_x[i,j] = a[i-1,j] + 0.5*(1.0 - cx)*ldelta_a[i-1,j]
        a_x.v(buf=1)[:, :] = a.ip(-1, buf=1) + 0.5*(1.0 - cx)*ldelta_ax.ip(-1, buf=1)

    # y-direction
    a_y = myg.scratch_array()

    # upwind
    if v < 0:
        # a_y[i,j] = a[i,j] - 0.5*(1.0 + cy)*ldelta_a[i,j]
        a_y.v(buf=1)[:, :] = a.v(buf=1) - 0.5*(1.0 + cy)*ldelta_ay.v(buf=1)
    else:
        # a_y[i,j] = a[i,j-1] + 0.5*(1.0 - cy)*ldelta_a[i,j-1]
        a_y.v(buf=1)[:, :] = a.jp(-1, buf=1) + 0.5*(1.0 - cy)*ldelta_ay.jp(-1, buf=1)

    # compute the transverse flux differences.  The flux is just (u a)
    # HOTF
    F_xt = u*a_x
    F_yt = v*a_y

    F_x = myg.scratch_array()
    F_y = myg.scratch_array()

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

    dtdx2 = 0.5*dt/myg.dx
    dtdy2 = 0.5*dt/myg.dy

    F_x.v(buf=1)[:, :] = u*(a_x.v(buf=1) -
                           dtdy2*(F_yt.ip_jp(mx, 1, buf=1) -
                                  F_yt.ip(mx, buf=1)))

    F_y.v(buf=1)[:, :] = v*(a_y.v(buf=1) -
                           dtdx2*(F_xt.ip_jp(1, my, buf=1) -
                                  F_xt.jp(my, buf=1)))

    return F_x, F_y
