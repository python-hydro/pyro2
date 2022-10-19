import numpy as np

import pyro.mesh.reconstruction as reconstruction


def unsplit_fluxes(my_data, rp, dt, scalar_name):
    """
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

    u = my_data.get_var("x-velocity")
    v = my_data.get_var("y-velocity")

    cx = u * dt / myg.dx
    cy = v * dt / myg.dy

    # --------------------------------------------------------------------------
    # monotonized central differences
    # --------------------------------------------------------------------------

    limiter = rp.get_param("advection.limiter")

    ldelta_ax = reconstruction.limit(a, myg, 1, limiter)
    ldelta_ay = reconstruction.limit(a, myg, 2, limiter)

    # upwind in x-direction
    a_x = myg.scratch_array()
    shift_x = my_data.get_var("x-shift").astype(int)

    for index, vel in np.ndenumerate(u.v(buf=1)):
        if vel < 0:
            a_x.v(buf=1)[index] = a.ip(shift_x.v(buf=1)[index], buf=1)[index] \
                - 0.5*(1.0 + cx.v(buf=1)[index]) \
                * ldelta_ax.ip(shift_x.v(buf=1)[index], buf=1)[index]
        else:
            a_x.v(buf=1)[index] = a.ip(shift_x.v(buf=1)[index], buf=1)[index] \
                + 0.5*(1.0 - cx.v(buf=1)[index]) \
                * ldelta_ax.ip(shift_x.v(buf=1)[index], buf=1)[index]

    # upwind in y-direction
    a_y = myg.scratch_array()
    shift_y = my_data.get_var("y-shift").astype(int)

    for index, vel in np.ndenumerate(v.v(buf=1)):
        if vel < 0:
            a_y.v(buf=1)[index] = a.jp(shift_y.v(buf=1)[index], buf=1)[index] \
                - 0.5*(1.0 + cy.v(buf=1)[index]) \
                * ldelta_ay.jp(shift_y.v(buf=1)[index], buf=1)[index]
        else:
            a_y.v(buf=1)[index] = a.jp(shift_y.v(buf=1)[index], buf=1)[index] \
                + 0.5*(1.0 - cy.v(buf=1)[index]) \
                * ldelta_ay.jp(shift_y.v(buf=1)[index], buf=1)[index]

    # compute the transverse flux differences.  The flux is just (u a)
    # HOTF
    F_xt = u * a_x
    F_yt = v * a_y

    F_x = myg.scratch_array()
    F_y = myg.scratch_array()

    # the zone where we grab the transverse flux derivative from
    # depends on the sign of the advective velocity
    dtdx2 = 0.5 * dt / myg.dx
    dtdy2 = 0.5 * dt / myg.dy

    for index, vel in np.ndenumerate(u.v(buf=1)):
        F_x.v(buf=1)[index] = vel * (a_x.v(buf=1)[index] - dtdy2 *
                (F_yt.ip_jp(shift_x.v(buf=1)[index], 1, buf=1)[index] -
                F_yt.ip(shift_x.v(buf=1)[index], buf=1)[index]))

    for index, vel in np.ndenumerate(v.v(buf=1)):
        F_y.v(buf=1)[index] = vel * (a_y.v(buf=1)[index] - dtdx2 *
                (F_xt.ip_jp(1, shift_y.v(buf=1)[index], buf=1)[index] -
                F_xt.jp(shift_y.v(buf=1)[index], buf=1)[index]))

    return F_x, F_y
