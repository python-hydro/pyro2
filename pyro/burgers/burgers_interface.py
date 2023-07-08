import numpy as np


def get_interface_states(grid, dt,
                         u, v,
                         ldelta_ux, ldelta_vx,
                         ldelta_uy, ldelta_vy):
    r"""
    Construct the interface states for the burgers equation:

    .. math::

       u_t  + u u_x  + v u_y  = 0
       v_t  + u v_x  + v v_y  = 0

    Compute the unsplit predictions of u and v on both the x- and
    y-interfaces.

    Parameters
    ----------
    grid : Grid2d
        The grid object
    dt : float
        The timestep
    u, v : ndarray
        x-velocity and y-velocity
    ldelta_ux, ldelta_uy: ndarray
        Limited slopes of the x-velocity in the x and y directions
    ldelta_vx, ldelta_vy: ndarray
        Limited slopes of the y-velocity in the x and y directions

    Returns
    -------
    out : ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray
        get the predictions of the left and right states of u and v on both the x- and
        y-interfaces interface states without the transverse terms.
    """

    u_xl = grid.scratch_array()
    u_xr = grid.scratch_array()
    u_yl = grid.scratch_array()
    u_yr = grid.scratch_array()

    v_xl = grid.scratch_array()
    v_xr = grid.scratch_array()
    v_yl = grid.scratch_array()
    v_yr = grid.scratch_array()

    # first predict u and v to both interfaces, considering only the normal
    # part of the predictor.  These are the 'hat' states.

    dtdx = dt / grid.dx
    dtdy = dt / grid.dy

    # u on x-edges
    u_xl.ip(1, buf=2)[:, :] = u.v(buf=2) + \
        0.5 * (1.0 - dtdx * u.v(buf=2)) * ldelta_ux.v(buf=2)
    u_xr.v(buf=2)[:, :] = u.v(buf=2) - \
        0.5 * (1.0 + dtdx * u.v(buf=2)) * ldelta_ux.v(buf=2)

    # v on x-edges
    v_xl.ip(1, buf=2)[:, :] = v.v(buf=2) + \
        0.5 * (1.0 - dtdx * u.v(buf=2)) * ldelta_vx.v(buf=2)
    v_xr.v(buf=2)[:, :] = v.v(buf=2) - \
        0.5 * (1.0 + dtdx * u.v(buf=2)) * ldelta_vx.v(buf=2)

    # u on y-edges
    u_yl.jp(1, buf=2)[:, :] = u.v(buf=2) + \
        0.5 * (1.0 - dtdy * v.v(buf=2)) * ldelta_uy.v(buf=2)
    u_yr.v(buf=2)[:, :] = u.v(buf=2) - \
        0.5 * (1.0 + dtdy * v.v(buf=2)) * ldelta_uy.v(buf=2)

    # v on y-edges
    v_yl.jp(1, buf=2)[:, :] = v.v(buf=2) + \
        0.5 * (1.0 - dtdy * v.v(buf=2)) * ldelta_vy.v(buf=2)
    v_yr.v(buf=2)[:, :] = v.v(buf=2) - \
        0.5 * (1.0 + dtdy * v.v(buf=2)) * ldelta_vy.v(buf=2)

    return u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr


def apply_transverse_corrections(grid, dt,
                                 u_xl, u_xr,
                                 u_yl, u_yr,
                                 v_xl, v_xr,
                                 v_yl, v_yr):
    r"""
    Parameters
    ----------
    grid : Grid2d
        The grid object
    dt : float
        The timestep
    u_xl, u_xr : ndarray ndarray
        left and right states of x-velocity in x-interface.
    u_yl, u_yr : ndarray ndarray
        left and right states of x-velocity in y-interface.
    v_xl, v_xr : ndarray ndarray
        left and right states of y-velocity in x-interface.
    v_yl, u_yr : ndarray ndarray
        left and right states of y-velocity in y-interface.
    Returns
    -------
    out : ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray
        correct the interface states of the left and right states of u and v on both the x- and
        y-interfaces interface states with the transverse terms.
    """

    dtdx = dt / grid.dx
    dtdy = dt / grid.dy

    # now get the normal advective velocities on the interfaces by solving
    # the Riemann problem.
    uhat_adv = riemann(grid, u_xl, u_xr)
    vhat_adv = riemann(grid, v_yl, v_yr)

    # now that we have the advective velocities, upwind the left and right
    # states using the appropriate advective velocity.

    # on the x-interfaces, we upwind based on uhat_adv
    u_xint = upwind(grid, u_xl, u_xr, uhat_adv)
    v_xint = upwind(grid, v_xl, v_xr, uhat_adv)

    # on the y-interfaces, we upwind based on vhat_adv
    u_yint = upwind(grid, u_yl, u_yr, vhat_adv)
    v_yint = upwind(grid, v_yl, v_yr, vhat_adv)

    # at this point, these states are the `hat' states -- they only
    # considered the normal to the interface portion of the predictor.

    ubar = grid.scratch_array()
    vbar = grid.scratch_array()

    ubar.v(buf=2)[:, :] = 0.5 * (uhat_adv.v(buf=2) + uhat_adv.ip(1, buf=2))
    vbar.v(buf=2)[:, :] = 0.5 * (vhat_adv.v(buf=2) + vhat_adv.jp(1, buf=2))

    # the transverse term for the u states on x-interfaces
    u_xl.ip(1, buf=2)[:, :] += -0.5 * dtdy * vbar.v(buf=2) * \
                               (u_yint.jp(1, buf=2) - u_yint.v(buf=2))
    u_xr.v(buf=2)[:, :] += -0.5 * dtdy * vbar.v(buf=2) * \
                           (u_yint.jp(1, buf=2) - u_yint.v(buf=2))

    # the transverse term for the v states on x-interfaces
    v_xl.ip(1, buf=2)[:, :] += -0.5 * dtdy * vbar.v(buf=2) * \
                               (v_yint.jp(1, buf=2) - v_yint.v(buf=2))
    v_xr.v(buf=2)[:, :] += -0.5 * dtdy * vbar.v(buf=2) * \
                           (v_yint.jp(1, buf=2) - v_yint.v(buf=2))

    # the transverse term for the v states on y-interfaces
    v_yl.jp(1, buf=2)[:, :] += -0.5 * dtdx * ubar.v(buf=2) * \
                               (v_xint.ip(1, buf=2) - v_xint.v(buf=2))
    v_yr.v(buf=2)[:, :] += -0.5 * dtdx * ubar.v(buf=2) * \
                           (v_xint.ip(1, buf=2) - v_xint.v(buf=2))

    # the transverse term for the u states on y-interfaces
    u_yl.jp(1, buf=2)[:, :] += -0.5 * dtdx * ubar.v(buf=2) * \
                               (u_xint.ip(1, buf=2) - u_xint.v(buf=2))
    u_yr.v(buf=2)[:, :] += -0.5 * dtdx * ubar.v(buf=2) * \
                           (u_xint.ip(1, buf=2) - u_xint.v(buf=2))

    return u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr


def construct_unsplit_fluxes(grid,
                             u_xl, u_xr,
                             u_yl, u_yr,
                             v_xl, v_xr,
                             v_yl, v_yr):
    r"""
    Construct the interface fluxes for the burgers equation:

    .. math::

       u_t  + u u_x  + v u_y  = 0
       v_t  + u v_x  + v v_y  = 0

    Parameters
    ----------
    grid : Grid2d
        The grid object
    u_xl, u_xr : ndarray ndarray
        left and right states of x-velocity in x-interface.
    u_yl, u_yr : ndarray ndarray
        left and right states of x-velocity in y-interface.
    v_xl, v_xr : ndarray ndarray
        left and right states of y-velocity in x-interface.
    v_yl, u_yr : ndarray ndarray
        left and right states of y-velocity in y-interface.
    -------
    Returns
    -------
    out : ndarray, ndarray, ndarray, ndarray
        The u,v fluxes on the x- and y-interfaces

    """

    # Solve for riemann problem for the second time

    # Get corrected normal advection velocity (MAC)

    u_MAC = riemann_and_upwind(grid, u_xl, u_xr)
    v_MAC = riemann_and_upwind(grid, v_yl, v_yr)

    # Upwind using the transverse corrected normal advective velocity

    ux = upwind(grid, u_xl, u_xr, u_MAC)
    vx = upwind(grid, v_xl, v_xr, u_MAC)

    uy = upwind(grid, u_yl, u_yr, v_MAC)
    vy = upwind(grid, v_yl, v_yr, v_MAC)

    # construct the flux

    fu_x = grid.scratch_array()
    fv_x = grid.scratch_array()
    fu_y = grid.scratch_array()
    fv_y = grid.scratch_array()

    fu_x.v(buf=2)[:, :] = 0.5 * ux.v(buf=2) * u_MAC.v(buf=2)
    fv_x.v(buf=2)[:, :] = 0.5 * vx.v(buf=2) * u_MAC.v(buf=2)

    fu_y.v(buf=2)[:, :] = 0.5 * uy.v(buf=2) * v_MAC.v(buf=2)
    fv_y.v(buf=2)[:, :] = 0.5 * vy.v(buf=2) * v_MAC.v(buf=2)

    return fu_x, fu_y, fv_x, fv_y


def upwind(grid, q_l, q_r, s):
    r"""
    upwind the left and right states based on the specified input
    velocity, s.  The resulting interface state is q_int

    Parameters
    ----------
    grid : Grid2d
        The grid object
    q_l, q_r : ndarray
        left and right states
    s : ndarray
        velocity

    Returns
    -------
    out : ndarray
        Upwinded state
    """

    q_int = grid.scratch_array()

    q_int.v(buf=2)[:, :] = np.where(s.v(buf=2) == 0.0,
                                    0.5 * (q_l.v(buf=2) + q_r.v(buf=2)),
                                    np.where(s.v(buf=2) > 0.0, q_l.v(buf=2), q_r.v(buf=2)))

    return q_int


def riemann(grid, q_l, q_r):
    """
    Solve the Burger's Riemann problem given the input left and right
    states and return the state on the interface.

    This uses the expressions from Almgren, Bell, and Szymczak 1996.

    Parameters
    ----------
    grid : Grid2d
        The grid object
    q_l, q_r : ndarray
        left and right states

    Returns
    -------
    out : ndarray
        Interface state
    """

    s = grid.scratch_array()

    s.v(buf=2)[:, :] = np.where(np.logical_and(q_l.v(buf=2) <= 0.0,
                                               q_r.v(buf=2) >= 0.0),
                                0.0,
                                np.where(np.logical_and(q_l.v(buf=2) > 0.0,
                                                        q_l.v(buf=2) + q_r.v(buf=2) > 0.0),
                                         q_l.v(buf=2), q_r.v(buf=2)))

    return s


def riemann_and_upwind(grid, q_l, q_r):
    r"""
    First solve the Riemann problem given q_l and q_r to give the
    velocity on the interface and: use this velocity to upwind to
    determine the state (q_l, q_r, or a mix) on the interface).

    This differs from upwind, above, in that we don't take in a
    velocity to upwind with).

    Parameters
    ----------
    grid : Grid2d
        The grid object
    q_l, q_r : ndarray
        left and right states

    Returns
    -------
    out : ndarray
        Upwinded state
    """

    s = riemann(grid, q_l, q_r)
    return upwind(grid, q_l, q_r, s)
