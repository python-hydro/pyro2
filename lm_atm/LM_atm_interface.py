import numpy as np
from numba import njit


@njit(cache=True)
def is_symmetric_pair(qx, qy, ng, nodal, sl, sr):

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    sym = 1

    if (not nodal):
        done = False
        for j in range(jlo, jhi):
            for i in range(nx / 2):
                il = ilo + i
                ir = ihi - i

                if (not sl(il, j) == sr(ir, j)):
                    sym = 0
                    done = True
                    break
            if done:
                break

    else:
        done = False

        for j in range(jlo, jhi):
            for i in range(nx / 2):
                il = ilo + i
                ir = ihi - i + 1

                if (not sl(il, j) == sr(ir, j)):
                    sym = 0
                    done = True
                    break

            if done:
                break

    return sym


@njit(cache=True)
def is_symmetric(qx, qy, ng, nodal, s):

    return is_symmetric_pair(qx, qy, ng, nodal, s, s)


@njit(cache=True)
def is_asymmetric_pair(qx, qy, ng, nodal, sl, sr):

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    asym = 1

    if (not nodal):
        done = False
        for j in range(jlo, jhi):
            for i in range(nx / 2):
                il = ilo + i
                ir = ihi - i

                if (not sl(il, j) == -sr(ir, j)):
                    asym = 0
                    done = True
                    break
            if done:
                break

    else:
        done = False

        for j in range(jlo, jhi):
            for i in range(nx / 2):
                il = ilo + i
                ir = ihi - i + 1

                # print *, il, ir, sl(il,j), -sr(ir,j)
                if (not sl(il, j) == -sr(ir, j)):
                    asym = 0
                    done = True
                    break

            if done:
                break

    return asym


@njit(cache=True)
def is_asymmetric(qx, qy, ng, nodal, s):

    return is_asymmetric_pair(qx, qy, ng, nodal, s, s)


@njit(cache=True)
def mac_vels(qx, qy, ng, dx, dy, dt,
             u, v,
             ldelta_ux, ldelta_vx,
             ldelta_uy, ldelta_vy,
             gradp_x, gradp_y,
             source):

    u_MAC = np.zeros((qx, qy))
    v_MAC = np.zeros((qx, qy))

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
    u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr = get_interface_states(qx, qy, ng, dx, dy, dt,
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
    riemann_and_upwind(qx, qy, ng, u_xl, u_xr, u_MAC)
    riemann_and_upwind(qx, qy, ng, v_yl, v_yr, v_MAC)

    # print *, 'checking U_MAC'
    # if (not is_asymmetric(qx, qy, ng, .true., u_MAC) == 1):
    #    stop 'u_MAC not asymmetric'
    #

    return u_MAC, v_MAC


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
@njit(cache=True)
def states(qx, qy, ng, dx, dy, dt,
           u, v,
           ldelta_ux, ldelta_vx,
           ldelta_uy, ldelta_vy,
           gradp_x, gradp_y,
           source,
           u_MAC, v_MAC):

    # this is similar to mac_vels, but it predicts the interface states
    # of both u and v on both interfaces, using the MAC velocities to
    # do the upwinding.

    u_xint = np.zeros((qx, qy))
    u_yint = np.zeros((qx, qy))
    v_xint = np.zeros((qx, qy))
    v_yint = np.zeros((qx, qy))

    # get the full u and v left and right states (including transverse
    # terms) on both the x- and y-interfaces
    u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr = get_interface_states(qx, qy, ng, dx, dy, dt,
                                                                          u, v,
                                                                          ldelta_ux, ldelta_vx,
                                                                          ldelta_uy, ldelta_vy,
                                                                          gradp_x, gradp_y,
                                                                          source)

    # upwind using the MAC velocity to determine which state exists on
    # the interface
    upwind(qx, qy, ng, u_xl, u_xr, u_MAC, u_xint)
    upwind(qx, qy, ng, v_xl, v_xr, u_MAC, v_xint)
    upwind(qx, qy, ng, u_yl, u_yr, v_MAC, u_yint)
    upwind(qx, qy, ng, v_yl, v_yr, v_MAC, v_yint)

    return u_xint, u_yint, v_xint, v_yint


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
@njit(cache=True)
def rho_states(qx, qy, ng, dx, dy, dt,
               rho, u_MAC, v_MAC,
               ldelta_rx, ldelta_ry):

    # this predicts rho to the interfaces.  We use the MAC velocities to do
    # the upwinding

    rho_xint = np.zeros((qx, qy))
    rho_yint = np.zeros((qx, qy))

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

    for j in range(jlo - 2, jhi + 2):
        for i in range(ilo - 2, ihi + 2):

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
    upwind(qx, qy, ng, rho_xl, rho_xr, u_MAC, rho_xint)
    upwind(qx, qy, ng, rho_yl, rho_yr, v_MAC, rho_yint)

    # now add the transverse term and the non-advective part of the normal
    # divergence
    for j in range(jlo - 2, jhi + 2):
        for i in range(ilo - 2, ihi + 2):

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
    upwind(qx, qy, ng, rho_xl, rho_xr, u_MAC, rho_xint)
    upwind(qx, qy, ng, rho_yl, rho_yr, v_MAC, rho_yint)

    return rho_xint, rho_yint


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
@njit(cache=True)
def get_interface_states(qx, qy, ng, dx, dy, dt,
                         u, v,
                         ldelta_ux, ldelta_vx,
                         ldelta_uy, ldelta_vy,
                         gradp_x, gradp_y,
                         source):

    # Compute the unsplit predictions of u and v on both the x- and
    # y-interfaces.  This includes the transverse terms.

    # note that the gradp_x, gradp_y should have any coefficients
    # already included (e.g. beta_0/rho)

    u_xl = np.zeros((qx, qy))
    u_xr = np.zeros((qx, qy))
    u_yl = np.zeros((qx, qy))
    u_yr = np.zeros((qx, qy))

    v_xl = np.zeros((qx, qy))
    v_xr = np.zeros((qx, qy))
    v_yl = np.zeros((qx, qy))
    v_yr = np.zeros((qx, qy))

    uhat_adv = np.zeros((qx, qy))
    vhat_adv = np.zeros((qx, qy))

    u_xint = np.zeros((qx, qy))
    u_yint = np.zeros((qx, qy))
    v_xint = np.zeros((qx, qy))
    v_yint = np.zeros((qx, qy))

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

    for j in range(jlo - 2, jhi + 2):
        for i in range(ilo - 2, ihi + 2):

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
    riemann(qx, qy, ng, u_xl, u_xr, uhat_adv)
    riemann(qx, qy, ng, v_yl, v_yr, vhat_adv)

    # now that we have the advective velocities, upwind the left and right
    # states using the appropriate advective velocity.

    # on the x-interfaces, we upwind based on uhat_adv
    upwind(qx, qy, ng, u_xl, u_xr, uhat_adv, u_xint)
    upwind(qx, qy, ng, v_xl, v_xr, uhat_adv, v_xint)

    # on the y-interfaces, we upwind based on vhat_adv
    upwind(qx, qy, ng, u_yl, u_yr, vhat_adv, u_yint)
    upwind(qx, qy, ng, v_yl, v_yr, vhat_adv, v_yint)

    # at this point, these states are the `hat' states -- they only
    # considered the normal to the interface portion of the predictor.

    # add the transverse flux differences to the preliminary interface states
    for j in range(jlo - 1, jhi + 1):
        for i in range(ilo - 1, ihi + 1):

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
def upwind(qx, qy, ng, q_l, q_r, s, q_int):

    # upwind the left and right states based on the specified input
    # velocity, s.  The resulting interface state is q_int

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    for j in range(jlo - 1, jhi + 2):
        for i in range(ilo - 1, ihi + 2):

            if (s[i, j] > 0.0):
                q_int[i, j] = q_l[i, j]
            elif (s[i, j] == 0.0):
                q_int[i, j] = 0.5 * (q_l[i, j] + q_r[i, j])
            else:
                q_int[i, j] = q_r[i, j]


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
@njit(cache=True)
def riemann(qx, qy, ng, q_l, q_r, s):

    # Solve the Burger's Riemann problem given the input left and right
    # states and return the state on the interface.
    #
    # This uses the expressions from Almgren, Bell, and Szymczak 1996.

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    for j in range(jlo - 1, jhi + 2):
        for i in range(ilo - 1, ihi + 2):

            if (q_l[i, j] > 0.0 and q_l[i, j] + q_r[i, j] > 0.0):
                s[i, j] = q_l[i, j]
            elif (q_l[i, j] <= 0.0 and q_r[i, j] >= 0.0):
                s[i, j] = 0.0
            else:
                s[i, j] = q_r[i, j]


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
@njit(cache=True)
def riemann_and_upwind(qx, qy, ng, q_l, q_r, q_int):

    # First solve the Riemann problem given q_l and q_r to give the
    # velocity on the interface and: use this velocity to upwind to
    # determine the state (q_l, q_r, or a mix) on the interface).
    #
    # This differs from upwind, above, in that we don't take in a
    # velocity to upwind with).

    s = np.zeros((qx, qy))

    riemann(qx, qy, ng, q_l, q_r, s)
    upwind(qx, qy, ng, q_l, q_r, s, q_int)
