import numpy as np
from numba import njit


@njit(cache=True)
def is_symmetric_pair(ng, nodal, sl, sr):
    r"""
    Are sl and sr symmetric about an axis parallel with the y-axis in the center of domain the x-direction?

    Parameters
    ----------
    ng : int
        The number of ghost cells
    nodal: bool
        Is the data nodal?
    sl, sr : ndarray
        The two arrays to be compared

    Returns
    -------
    out : int
        Are they symmetric? (1 = yes, 0 = no)
    """

    qx, qy = sl.shape

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    sym = 1

    if (not nodal):
        done = False
        for i in range(nx / 2):
            il = ilo + i
            ir = ihi - i

            for j in range(jlo, jhi):
                if (sl[il, j] != sr[ir, j]):
                    sym = 0
                    done = True
                    break
            if done:
                break

    else:
        done = False

        for i in range(nx / 2):
            il = ilo + i
            ir = ihi - i + 1

            for j in range(jlo, jhi):
                if (sl[il, j] != sr[ir, j]):
                    sym = 0
                    done = True
                    break

            if done:
                break

    return sym


@njit(cache=True)
def is_symmetric(ng, nodal, s):
    r"""
    Is the left half of s the mirror image of the right half?

    Parameters
    ----------
    ng : int
        The number of ghost cells
    nodal: bool
        Is the data nodal?
    s : ndarray
        The array to be compared

    Returns
    -------
    out : int
        Is it symmetric? (1 = yes, 0 = no)
    """

    return is_symmetric_pair(ng, nodal, s, s)


@njit(cache=True)
def is_asymmetric_pair(ng, nodal, sl, sr):
    r"""
    Are sl and sr asymmetric about an axis parallel with the y-axis in the center of domain the x-direction?

    Parameters
    ----------
    ng : int
        The number of ghost cells
    nodal: bool
        Is the data nodal?
    sl, sr : ndarray
        The two arrays to be compared

    Returns
    -------
    out : int
        Are they asymmetric? (1 = yes, 0 = no)
    """

    qx, qy = sl.shape

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    asym = 1

    if (not nodal):
        done = False
        for i in range(nx / 2):
            for j in range(jlo, jhi):
                il = ilo + i
                ir = ihi - i

                if (not sl[il, j] == -sr[ir, j]):
                    asym = 0
                    done = True
                    break
            if done:
                break

    else:
        done = False

        for i in range(nx / 2):
            for j in range(jlo, jhi):
                il = ilo + i
                ir = ihi - i + 1

                # print *, il, ir, sl(il,j), -sr(ir,j)
                if (not sl[il, j] == -sr[ir, j]):
                    asym = 0
                    done = True
                    break

            if done:
                break

    return asym


@njit(cache=True)
def is_asymmetric(ng, nodal, s):
    """
    Is the left half of s asymmetric to the right half?

    Parameters
    ----------
    ng : int
        The number of ghost cells
    nodal: bool
        Is the data nodal?
    s : ndarray
        The array to be compared

    Returns
    -------
    out : int
        Is it asymmetric? (1 = yes, 0 = no)
    """

    return is_asymmetric_pair(ng, nodal, s, s)


@njit(cache=True)
def mac_vels(ng, dx, dy, dt,
             u, v,
             ldelta_ux, ldelta_vx,
             ldelta_uy, ldelta_vy,
             gradp_x, gradp_y,
             source):
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
    source : ndarray
        Source terms

    Returns
    -------
    out : ndarray, ndarray
        MAC velocities in the x and y directions
    """

    # assertions
    # print *, "checking ldelta_ux"
    # if (not is_asymmetric(qx, qy, ng, .false., ldelta_ux) == 1):
    #    stop 'ldelta_ux not asymmetric'
    #

    # print *, "checking ldelta_uy"
    # if (not is_symmetric(qx, qy, ng, .false., ldelta_uy) == 1):
    #    stop 'ldelta_uy not symmetric'
    #

    # print *, "checking ldelta_vx"
    # if (not is_symmetric(qx, qy, ng, .false., ldelta_vx) == 1):
    #    stop 'ldelta_vx not symmetric'
    #

    # print *, "checking ldelta_vy"
    # if (not is_symmetric(qx, qy, ng, .false., ldelta_vy) == 1):
    #    stop 'ldelta_vy not symmetric'
    #

    # print *, "checking gradp_x"
    # if (not is_asymmetric(qx, qy, ng, .false., gradp_x) == 1):
    #    stop 'gradp_x not asymmetric'
    #

    # get the full u and v left and right states (including transverse
    # terms) on both the x- and y-interfaces
    u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr = get_interface_states(ng, dx, dy, dt,
                                                                          u, v,
                                                                          ldelta_ux, ldelta_vx,
                                                                          ldelta_uy, ldelta_vy,
                                                                          gradp_x, gradp_y,
                                                                          source)

    # print *, 'checking u_xl'
    # if (not is_asymmetric_pair(qx, qy, ng, .true., u_xl, u_xr) == 1):
    #    stop 'u_xl/r not asymmetric'
    #

    # Riemann problem -- this follows Burger's equation.  We don't use
    # any input velocity for the upwinding.  Also, we only care about
    # the normal states here (u on x and v on y)
    u_MAC = riemann_and_upwind(ng, u_xl, u_xr)
    v_MAC = riemann_and_upwind(ng, v_yl, v_yr)

    # print *, 'checking U_MAC'
    # if (not is_asymmetric(qx, qy, ng, .true., u_MAC) == 1):
    #    stop 'u_MAC not asymmetric'
    #

    return u_MAC, v_MAC


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
@njit(cache=True)
def states(ng, dx, dy, dt,
           u, v,
           ldelta_ux, ldelta_vx,
           ldelta_uy, ldelta_vy,
           gradp_x, gradp_y,
           source,
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
    source : ndarray
        Source terms
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
                                                                          gradp_x, gradp_y,
                                                                          source)

    # upwind using the MAC velocity to determine which state exists on
    # the interface
    u_xint = upwind(ng, u_xl, u_xr, u_MAC)
    v_xint = upwind(ng, v_xl, v_xr, u_MAC)
    u_yint = upwind(ng, u_yl, u_yr, v_MAC)
    v_yint = upwind(ng, v_yl, v_yr, v_MAC)

    return u_xint, v_xint, u_yint, v_yint


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
@njit(cache=True)
def rho_states(ng, dx, dy, dt,
               rho, u_MAC, v_MAC,
               ldelta_rx, ldelta_ry):
    r"""
    This predicts rho to the interfaces.  We use the MAC velocities to do
    the upwinding

    Parameters
    ----------
    ng : int
        The number of ghost cells
    dx, dy : float
        The cell spacings
    rho : ndarray
        density
    u_MAC, v_MAC : ndarray
        MAC velocities in the x and y directions
    ldelta_rx, ldelta_ry: ndarray
        Limited slopes of the density in the x and y directions

    Returns
    -------
    out : ndarray, ndarray
        rho predicted to the interfaces
    """

    qx, qy = rho.shape

    rho_xl = np.zeros((qx, qy))
    rho_xr = np.zeros((qx, qy))
    rho_yl = np.zeros((qx, qy))
    rho_yr = np.zeros((qx, qy))

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    dtdx = dt / dx
    dtdy = dt / dy

    for i in range(ilo - 2, ihi + 2):
        for j in range(jlo - 2, jhi + 2):

            # u on x-edges
            rho_xl[i + 1, j] = rho[i, j] + 0.5 * \
                (1.0 - dtdx * u_MAC[i + 1, j]) * ldelta_rx[i, j]
            rho_xr[i, j] = rho[i, j] - 0.5 * \
                (1.0 + dtdx * u_MAC[i, j]) * ldelta_rx[i, j]

            # u on y-edges
            rho_yl[i, j + 1] = rho[i, j] + 0.5 * \
                (1.0 - dtdy * v_MAC[i, j + 1]) * ldelta_ry[i, j]
            rho_yr[i, j] = rho[i, j] - 0.5 * \
                (1.0 + dtdy * v_MAC[i, j]) * ldelta_ry[i, j]

    # we upwind based on the MAC velocities
    rho_xint = upwind(ng, rho_xl, rho_xr, u_MAC)
    rho_yint = upwind(ng, rho_yl, rho_yr, v_MAC)

    # now add the transverse term and the non-advective part of the normal
    # divergence
    for i in range(ilo - 2, ihi + 2):
        for j in range(jlo - 2, jhi + 2):

            u_x = (u_MAC[i + 1, j] - u_MAC[i, j]) / dx
            v_y = (v_MAC[i, j + 1] - v_MAC[i, j]) / dy

            # (rho v)_y is the transverse term for the x-interfaces
            # rho u_x is the non-advective piece for the x-interfaces
            rhov_y = (rho_yint[i, j + 1] * v_MAC[i, j + 1] -
                      rho_yint[i, j] * v_MAC[i, j]) / dy

            rho_xl[i + 1, j] = rho_xl[i + 1, j] - \
                0.5 * dt * (rhov_y + rho[i, j] * u_x)
            rho_xr[i, j] = rho_xr[i, j] - 0.5 * dt * (rhov_y + rho[i, j] * u_x)

            # (rho u)_x is the transverse term for the y-interfaces
            # rho v_y is the non-advective piece for the y-interfaces
            rhou_x = (rho_xint[i + 1, j] * u_MAC[i + 1, j] -
                      rho_xint[i, j] * u_MAC[i, j]) / dx

            rho_yl[i, j + 1] = rho_yl[i, j + 1] - \
                0.5 * dt * (rhou_x + rho[i, j] * v_y)
            rho_yr[i, j] = rho_yr[i, j] - 0.5 * dt * (rhou_x + rho[i, j] * v_y)

    # finally upwind the full states
    rho_xint = upwind(ng, rho_xl, rho_xr, u_MAC)
    rho_yint = upwind(ng, rho_yl, rho_yr, v_MAC)

    return rho_xint, rho_yint


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
@njit(cache=True)
def get_interface_states(ng, dx, dy, dt,
                         u, v,
                         ldelta_ux, ldelta_vx,
                         ldelta_uy, ldelta_vy,
                         gradp_x, gradp_y,
                         source):
    r"""
    Compute the unsplit predictions of u and v on both the x- and
    y-interfaces.  This includes the transverse terms.

    Note that the ``gradp_x``, ``gradp_y`` should have any coefficients
    already included (e.g. :math:`\beta_0/\rho`)

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
    source : ndarray
        Source terms

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

    # print *, 'checking u_xl in states'
    # if (not is_asymmetric_pair(qx, qy, ng, .true., u_xl, u_xr) == 1):
    #    stop 'u_xl/r not asymmetric'
    #

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
    for i in range(ilo - 1, ihi + 1):
        for j in range(jlo - 1, jhi + 1):

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

            v_xl[i + 1, j] = v_xl[i + 1, j] - 0.5 * dtdy * vv_y - \
                0.5 * dt * gradp_y[i, j] + 0.5 * dt * source[i, j]
            v_xr[i, j] = v_xr[i, j] - 0.5 * dtdy * vv_y - 0.5 * \
                dt * gradp_y[i, j] + 0.5 * dt * source[i, j]

            # u dv/dx is the transverse term for the v states on y-interfaces
            uv_x = ubar * (v_xint[i + 1, j] - v_xint[i, j])

            v_yl[i, j + 1] = v_yl[i, j + 1] - 0.5 * dtdx * uv_x - \
                0.5 * dt * gradp_y[i, j] + 0.5 * dt * source[i, j]
            v_yr[i, j] = v_yr[i, j] - 0.5 * dtdx * uv_x - 0.5 * \
                dt * gradp_y[i, j] + 0.5 * dt * source[i, j]

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
    q_int : ndarray
        Upwinded state
    """

    qx, qy = s.shape

    q_int = np.zeros_like(s)

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    for i in range(ilo - 1, ihi + 2):
        for j in range(jlo - 1, jhi + 2):

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

    s = np.zeros((qx, qy))

    for i in range(ilo - 1, ihi + 2):
        for j in range(jlo - 1, jhi + 2):

            if ((q_l[i, j] > 0.0) and (q_l[i, j] + q_r[i, j] > 0.0)):
                s[i, j] = q_l[i, j]
            elif ((q_l[i, j] <= 0.0) and (q_r[i, j] >= 0.0)):
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
