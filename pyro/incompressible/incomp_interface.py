import numpy as np

from pyro.burgers import burgers_interface


def mac_vels(grid,  dt,
             u, v,
             ldelta_ux, ldelta_vx,
             ldelta_uy, ldelta_vy,
             gradp_x, gradp_y):
    r"""
    Calculate the MAC velocities in the x and y directions.

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
    gradp_x, gradp_y : ndarray
        Pressure gradients in the x and y directions

    Returns
    -------
    out : ndarray, ndarray
        MAC velocities in the x and y directions
    """

    # get the full u and v left and right states (including transverse
    # terms) on both the x- and y-interfaces
    # pylint: disable-next=unused-variable
    u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr = get_interface_states(grid, dt,
                                                                          u, v,
                                                                          ldelta_ux, ldelta_vx,
                                                                          ldelta_uy, ldelta_vy,
                                                                          gradp_x, gradp_y)

    # Riemann problem -- this follows Burger's equation.  We don't use
    # any input velocity for the upwinding.  Also, we only care about
    # the normal states here (u on x and v on y)
    u_MAC = riemann_and_upwind(grid, u_xl, u_xr)
    v_MAC = riemann_and_upwind(grid, v_yl, v_yr)

    return u_MAC, v_MAC


def states(grid, dt,
           u, v,
           ldelta_ux, ldelta_vx,
           ldelta_uy, ldelta_vy,
           gradp_x, gradp_y,
           u_MAC, v_MAC):
    r"""
    This is similar to ``mac_vels``, but it predicts the interface states
    of both u and v on both interfaces, using the MAC velocities to
    do the upwinding.

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
    gradp_x, gradp_y : ndarray
        Pressure gradients in the x and y directions
    u_MAC, v_MAC : ndarray
        MAC velocities in the x and y directions

    Returns
    -------
    out : ndarray, ndarray, ndarray, ndarray
        x and y velocities predicted to the interfaces
    """

    # get the full u and v left and right states (including transverse
    # terms) on both the x- and y-interfaces
    u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr = get_interface_states(grid, dt,
                                                                          u, v,
                                                                          ldelta_ux, ldelta_vx,
                                                                          ldelta_uy, ldelta_vy,
                                                                          gradp_x, gradp_y)

    # upwind using the MAC velocity to determine which state exists on
    # the interface
    u_xint = upwind(grid, u_xl, u_xr, u_MAC)
    v_xint = upwind(grid, v_xl, v_xr, u_MAC)
    u_yint = upwind(grid, u_yl, u_yr, v_MAC)
    v_yint = upwind(grid, v_yl, v_yr, v_MAC)

    return u_xint, v_xint, u_yint, v_yint


def get_interface_states(grid, dt,
                         u, v,
                         ldelta_ux, ldelta_vx,
                         ldelta_uy, ldelta_vy,
                         gradp_x, gradp_y):
    r"""
    Compute the unsplit predictions of u and v on both the x- and
    y-interfaces.  This includes the transverse terms.

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
    gradp_x, gradp_y : ndarray
        Pressure gradients in the x and y directions

    Returns
    -------
    out : ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray
        unsplit predictions of u and v on both the x- and
        y-interfaces
    """

    # Get the left and right interface states

    u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr = burgers_interface.get_interface_states(grid, dt,
                                                                                      u, v,
                                                                                 ldelta_ux, ldelta_vx,
                                                                                 ldelta_uy, ldelta_vy)

    # Apply pressure gradient correction terms

    # transverse term for the u states on x-interfaces
    u_xl.ip(1, buf=2)[:, :] += - 0.5 * dt * gradp_x.v(buf=2)
    u_xr.v(buf=2)[:, :] += - 0.5 * dt * gradp_x.v(buf=2)

    # transverse term for the v states on x-interfaces
    v_xl.ip(1, buf=2)[:, :] += - 0.5 * dt * gradp_y.v(buf=2)
    v_xr.v(buf=2)[:, :] += - 0.5 * dt * gradp_y.v(buf=2)

    # transverse term for the v states on y-interfaces
    v_yl.jp(1, buf=2)[:, :] += - 0.5 * dt * gradp_y.v(buf=2)
    v_yr.v(buf=2)[:, :] += - 0.5 * dt * gradp_y.v(buf=2)

    # transverse term for the u states on y-interfaces
    u_yl.jp(1, buf=2)[:, :] += - 0.5 * dt * gradp_x.v(buf=2)
    u_yr.v(buf=2)[:, :] += - 0.5 * dt * gradp_x.v(buf=2)

    return u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr


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

    q_int.v(buf=1)[:, :] = np.where(s.v(buf=1) == 0.0,
                                    0.5 * (q_l.v(buf=1) + q_r.v(buf=1)),
                                    np.where(s.v(buf=1) > 0.0, q_l.v(buf=1), q_r.v(buf=1)))

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

    s.v(buf=1)[:, :] = np.where(np.logical_and(q_l.v(buf=1) <= 0.0,
                                               q_r.v(buf=1) >= 0.0),
                                0.0,
                                np.where(np.logical_and(q_l.v(buf=1) > 0.0,
                                                        q_l.v(buf=1) + q_r.v(buf=1) > 0.0),
                                         q_l.v(buf=1), q_r.v(buf=1)))

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
