from pyro.mesh.reconstruction import ppm_reconstruction


def unsplit_fluxes(data, rp, dt, scalar_name):
    r"""
    Construct the fluxes through the interfaces for the linear advection
    equation:

    .. math::

       a_t  + u a_x  + v a_y  = 0

    We use a second-order (piecewise parabolic) unsplit Godunov method
    (following Colella 1990, and Colella & Woodward 1984).

    In the pure advection case, there is no Riemann problem we need to
    solve -- we just simply do upwinding.  So there is only one 'state'
    at each interface, and the zone the information comes from depends
    on the sign of the velocity.

    Our convection is that the fluxes are going to be defined on the
    left edge of the computational zones::

               |                                       |
               |                                       |
       --------+------------------+--------------------+--------
               |                  i                    |
               |                                       |
               |a_R,i-1/2        a_i        a_{L,i+1/2}|

    a_{R,i-1/2} and a_{l,i+1/2} are computed using the information in
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

    myg = data.grid

    a = data.get_var(scalar_name)

    # get the advection velocities
    u = rp.get_param("advection.u")
    v = rp.get_param("advection.v")

    cx = u*dt/myg.dx
    cy = v*dt/myg.dy

    # --------------------------------------------------------------------------
    # monotonized central differences
    # --------------------------------------------------------------------------

    delta_ax = myg.scratch_array()
    a6x = myg.scratch_array()
    delta_ay = myg.scratch_array()
    a6y = myg.scratch_array()

    # upwind
    a_x = myg.scratch_array()

    ar, al = ppm_reconstruction(a, myg, idir=1)
    delta_ax.v(buf=1)[:, :] = al.v(buf=1) - ar.v(buf=1)
    a6x.v(buf=1)[:, :] = 6.0 * (a.v(buf=1) - 0.5*(ar.v(buf=1) + al.v(buf=1)))

    if u < 0:
        cx = -cx
        # a_x[i,j] = a[i,j] - 0.5*(1.0 + cx)*ldelta_a[i,j]
        a_x.v(buf=1)[:, :] = ar.v(buf=1) + 0.5 * cx * (delta_ax.v(buf=1) +
                                (1 - 2.0 * cx / 3.0) * a6x.v(buf=1))
    else:
        # a_x[i,j] = a[i-1,j] + 0.5*(1.0 - cx)*ldelta_a[i-1,j]
        a_x.v(buf=1)[:, :] = al.ip(-1, buf=1) - 0.5 * cx * (delta_ax.ip(-1, buf=1) -
                                (1 - 2.0 * cx / 3.0) * a6x.ip(-1, buf=1))

    # y-direction
    a_y = myg.scratch_array()

    ar, al = ppm_reconstruction(a, myg, idir=2)
    delta_ay.v(buf=1)[:, :] = al.v(buf=1) - ar.v(buf=1)
    a6y.v(buf=1)[:, :] = 6.0 * (a.v(buf=1) - 0.5*(ar.v(buf=1) + al.v(buf=1)))

    # upwind
    if v < 0:
        cy = -cy
        # a_y[i,j] = a[i,j] - 0.5*(1.0 + cy)*ldelta_a[i,j]
        a_y.v(buf=1)[:, :] = ar.v(buf=1) + 0.5 * cy * (delta_ay.v(buf=1) +
                                (1 - 2.0 * cy / 3.0) * a6y.v(buf=1))
    else:
        # a_y[i,j] = a[i,j-1] + 0.5*(1.0 - cy)*ldelta_a[i,j-1]
        a_y.v(buf=1)[:, :] = al.jp(-1, buf=1) - 0.5 * cy * (delta_ay.jp(-1, buf=1) -
                                (1 - 2.0 * cy / 3.0) * a6y.jp(-1, buf=1))

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
