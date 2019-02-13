import numpy as np
from numba import njit


@njit(cache=True)
def mac_vels(ng, dx, dy, dt,
             u, v,
             ldelta_ux, ldelta_vx,
             ldelta_uy, ldelta_vy,
             gradp_x, gradp_y):
    r"""
    Calculate the MAC velocities in the x and y directions.

    Parameters
    ----------
    ng : int
        The number of ghost cells
    dx, dy : float
        The cell spacings
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
    u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr = get_interface_states(ng, dx, dy, dt,
                                                                          u, v,
                                                                          ldelta_ux, ldelta_vx,
                                                                          ldelta_uy, ldelta_vy,
                                                                          gradp_x, gradp_y)

    # Riemann problem -- this follows Burger's equation.  We don't use
    # any input velocity for the upwinding.  Also, we only care about
    # the normal states here (u on x and v on y)
    u_MAC = riemann_and_upwind(ng, u_xl, u_xr)
    v_MAC = riemann_and_upwind(ng, v_yl, v_yr)

    return u_MAC, v_MAC


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
@njit(cache=True)
def states(ng, dx, dy, dt,
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
    ng : int
        The number of ghost cells
    dx, dy : float
        The cell spacings
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
    u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr = get_interface_states(ng, dx, dy, dt,
                                                                          u, v,
                                                                          ldelta_ux, ldelta_vx,
                                                                          ldelta_uy, ldelta_vy,
                                                                          gradp_x, gradp_y)

    # upwind using the MAC velocity to determine which state exists on
    # the interface
    u_xint = upwind(ng, u_xl, u_xr, u_MAC)
    v_xint = upwind(ng, v_xl, v_xr, u_MAC)
    u_yint = upwind(ng, u_yl, u_yr, v_MAC)
    v_yint = upwind(ng, v_yl, v_yr, v_MAC)

    return u_xint, v_xint, u_yint, v_yint


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
@njit(cache=True)
def get_interface_states(ng, dx, dy, dt,
                         u, v,
                         ldelta_ux, ldelta_vx,
                         ldelta_uy, ldelta_vy,
                         gradp_x, gradp_y):
    r"""
    Compute the unsplit predictions of u and v on both the x- and
    y-interfaces.  This includes the transverse terms.

    Parameters
    ----------
    ng : int
        The number of ghost cells
    dx, dy : float
        The cell spacings
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

    qx, qy = u.shape

    u_xl = np.zeros((qx, qy))
    u_xr = np.zeros((qx, qy))
    u_yl = np.zeros((qx, qy))
    u_yr = np.zeros((qx, qy))

    v_xl = np.zeros((qx, qy))
    v_xr = np.zeros((qx, qy))
    v_yl = np.zeros((qx, qy))
    v_yr = np.zeros((qx, qy))

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    # first predict u and v to both interfaces, considering only the normal
    # part of the predictor.  These are the 'hat' states.

    dtdx = dt / dx
    dtdy = dt / dy

    for i in range(ilo - 2, ihi + 2):
        for j in range(jlo - 2, jhi + 2):

            # u on x-edges
            u_xl[i + 1, j] = u[i, j] + 0.5 * \
                (1.0 - dtdx * u[i, j]) * ldelta_ux[i, j]
            u_xr[i, j] = u[i, j] - 0.5 * \
                (1.0 + dtdx * u[i, j]) * ldelta_ux[i, j]

            # v on x-edges
            v_xl[i + 1, j] = v[i, j] + 0.5 * \
                (1.0 - dtdx * u[i, j]) * ldelta_vx[i, j]
            v_xr[i, j] = v[i, j] - 0.5 * \
                (1.0 + dtdx * u[i, j]) * ldelta_vx[i, j]

            # u on y-edges
            u_yl[i, j + 1] = u[i, j] + 0.5 * \
                (1.0 - dtdy * v[i, j]) * ldelta_uy[i, j]
            u_yr[i, j] = u[i, j] - 0.5 * \
                (1.0 + dtdy * v[i, j]) * ldelta_uy[i, j]

            # v on y-edges
            v_yl[i, j + 1] = v[i, j] + 0.5 * \
                (1.0 - dtdy * v[i, j]) * ldelta_vy[i, j]
            v_yr[i, j] = v[i, j] - 0.5 * \
                (1.0 + dtdy * v[i, j]) * ldelta_vy[i, j]

    # now get the normal advective velocities on the interfaces by solving
    # the Riemann problem.
    uhat_adv = riemann(ng, u_xl, u_xr)
    vhat_adv = riemann(ng, v_yl, v_yr)

    # now that we have the advective velocities, upwind the left and right
    # states using the appropriate advective velocity.

    # on the x-interfaces, we upwind based on uhat_adv
    u_xint = upwind(ng, u_xl, u_xr, uhat_adv)
    v_xint = upwind(ng, v_xl, v_xr, uhat_adv)

    # on the y-interfaces, we upwind based on vhat_adv
    u_yint = upwind(ng, u_yl, u_yr, vhat_adv)
    v_yint = upwind(ng, v_yl, v_yr, vhat_adv)

    # at this point, these states are the `hat' states -- they only
    # considered the normal to the interface portion of the predictor.

    # add the transverse flux differences to the preliminary interface states
    for i in range(ilo - 2, ihi + 2):
        for j in range(jlo - 2, jhi + 2):

            ubar = 0.5 * (uhat_adv[i, j] + uhat_adv[i + 1, j])
            vbar = 0.5 * (vhat_adv[i, j] + vhat_adv[i, j + 1])

            # v du/dy is the transerse term for the u states on x-interfaces
            vu_y = vbar * (u_yint[i, j + 1] - u_yint[i, j])

            u_xl[i + 1, j] = u_xl[i + 1, j] - 0.5 * \
                dtdy * vu_y - 0.5 * dt * gradp_x[i, j]
            u_xr[i, j] = u_xr[i, j] - 0.5 * dtdy * \
                vu_y - 0.5 * dt * gradp_x[i, j]

            # v dv/dy is the transverse term for the v states on x-interfaces
            vv_y = vbar * (v_yint[i, j + 1] - v_yint[i, j])

            v_xl[i + 1, j] = v_xl[i + 1, j] - 0.5 * \
                dtdy * vv_y - 0.5 * dt * gradp_y[i, j]
            v_xr[i, j] = v_xr[i, j] - 0.5 * dtdy * \
                vv_y - 0.5 * dt * gradp_y[i, j]

            # u dv/dx is the transverse term for the v states on y-interfaces
            uv_x = ubar * (v_xint[i + 1, j] - v_xint[i, j])

            v_yl[i, j + 1] = v_yl[i, j + 1] - 0.5 * \
                dtdx * uv_x - 0.5 * dt * gradp_y[i, j]
            v_yr[i, j] = v_yr[i, j] - 0.5 * dtdx * \
                uv_x - 0.5 * dt * gradp_y[i, j]

            # u du/dx is the transverse term for the u states on y-interfaces
            uu_x = ubar * (u_xint[i + 1, j] - u_xint[i, j])

            u_yl[i, j + 1] = u_yl[i, j + 1] - 0.5 * \
                dtdx * uu_x - 0.5 * dt * gradp_x[i, j]
            u_yr[i, j] = u_yr[i, j] - 0.5 * dtdx * \
                uu_x - 0.5 * dt * gradp_x[i, j]

    return u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
@njit(cache=True)
def upwind(ng, q_l, q_r, s):
    r"""
    upwind the left and right states based on the specified input
    velocity, s.  The resulting interface state is q_int

    Parameters
    ----------
    ng : int
        The number of ghost cells
    q_l, q_r : ndarray
        left and right states
    s : ndarray
        velocity

    Returns
    -------
    out : ndarray
        Upwinded state
    """

    qx, qy = s.shape

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    q_int = np.zeros_like(s)

    for i in range(ilo - 1, ihi + 1):
        for j in range(jlo - 1, jhi + 1):

            if (s[i, j] > 0.0):
                q_int[i, j] = q_l[i, j]
            elif (s[i, j] == 0.0):
                q_int[i, j] = 0.5 * (q_l[i, j] + q_r[i, j])
            else:
                q_int[i, j] = q_r[i, j]

    return q_int


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
@njit(cache=True)
def riemann(ng, q_l, q_r):
    """
    Solve the Burger's Riemann problem given the input left and right
    states and return the state on the interface.

    This uses the expressions from Almgren, Bell, and Szymczak 1996.

    Parameters
    ----------
    ng : int
        The number of ghost cells
    q_l, q_r : ndarray
        left and right states

    Returns
    -------
    out : ndarray
        Interface state
    """

    qx, qy = q_l.shape

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    s = np.zeros_like(q_l)

    for i in range(ilo - 1, ihi + 1):
        for j in range(jlo - 1, jhi + 1):

            if (q_l[i, j] > 0.0 and q_l[i, j] + q_r[i, j] > 0.0):
                s[i, j] = q_l[i, j]
            elif (q_l[i, j] <= 0.0 and q_r[i, j] >= 0.0):
                s[i, j] = 0.0
            else:
                s[i, j] = q_r[i, j]

    return s


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
@njit(cache=True)
def riemann_and_upwind(ng, q_l, q_r):
    r"""
    First solve the Riemann problem given q_l and q_r to give the
    velocity on the interface and: use this velocity to upwind to
    determine the state (q_l, q_r, or a mix) on the interface).

    This differs from upwind, above, in that we don't take in a
    velocity to upwind with).

    Parameters
    ----------
    ng : int
        The number of ghost cells
    q_l, q_r : ndarray
        left and right states

    Returns
    -------
    out : ndarray
        Upwinded state
    """

    s = riemann(ng, q_l, q_r)
    return upwind(ng, q_l, q_r, s)
