import pyro.incompressible.incomp_interface as incomp_interface


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

    Returns
    -------
    out : ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray
        unsplit predictions of the left and right states of u and v on both the x- and
        y-interfaces
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

    # now get the normal advective velocities on the interfaces by solving
    # the Riemann problem.
    uhat_adv = incomp_interface.riemann(grid, u_xl, u_xr)
    vhat_adv = incomp_interface.riemann(grid, v_yl, v_yr)

    # now that we have the advective velocities, upwind the left and right
    # states using the appropriate advective velocity.

    # on the x-interfaces, we upwind based on uhat_adv
    u_xint = incomp_interface.upwind(grid, u_xl, u_xr, uhat_adv)
    v_xint = incomp_interface.upwind(grid, v_xl, v_xr, uhat_adv)

    # on the y-interfaces, we upwind based on vhat_adv
    u_yint = incomp_interface.upwind(grid, u_yl, u_yr, vhat_adv)
    v_yint = incomp_interface.upwind(grid, v_yl, v_yr, vhat_adv)

    # at this point, these states are the `hat' states -- they only
    # considered the normal to the interface portion of the predictor.

    ubar = grid.scratch_array()
    vbar = grid.scratch_array()

    ubar.v(buf=2)[:, :] = 0.5 * (uhat_adv.v(buf=2) + uhat_adv.ip(1, buf=2))
    vbar.v(buf=2)[:, :] = 0.5 * (vhat_adv.v(buf=2) + vhat_adv.jp(1, buf=2))

    # the transverse term for the u states on x-interfaces
    u_xl.ip(1, buf=2)[:, :] += -0.5 * dtdy * vbar.v(buf=2) * (u_yint.jp(1, buf=2) - u_yint.v(buf=2))
    u_xr.v(buf=2)[:, :] += -0.5 * dtdy * vbar.v(buf=2) * (u_yint.jp(1, buf=2) - u_yint.v(buf=2))

    # the transverse term for the v states on x-interfaces
    v_xl.ip(1, buf=2)[:, :] += -0.5 * dtdy * vbar.v(buf=2) * (v_yint.jp(1, buf=2) - v_yint.v(buf=2))
    v_xr.v(buf=2)[:, :] += -0.5 * dtdy * vbar.v(buf=2) * (v_yint.jp(1, buf=2) - v_yint.v(buf=2))

    # the transverse term for the v states on y-interfaces
    v_yl.jp(1, buf=2)[:, :] += -0.5 * dtdx * ubar.v(buf=2) * (v_xint.ip(1, buf=2) - v_xint.v(buf=2))
    v_yr.v(buf=2)[:, :] += -0.5 * dtdx * ubar.v(buf=2) * (v_xint.ip(1, buf=2) - v_xint.v(buf=2))

    # the transverse term for the u states on y-interfaces
    u_yl.jp(1, buf=2)[:, :] += -0.5 * dtdx * ubar.v(buf=2) * (u_xint.ip(1, buf=2) - u_xint.v(buf=2))
    u_yr.v(buf=2)[:, :] += -0.5 * dtdx * ubar.v(buf=2) * (u_xint.ip(1, buf=2) - u_xint.v(buf=2))

    return u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr


def unsplit_fluxes(grid, dt,
                   u, v,
                   ldelta_ux, ldelta_vx,
                   ldelta_uy, ldelta_vy):
    r"""
    Construct the interface fluxes for the burgers equation:

    .. math::

       u_t  + u u_x  + v u_y  = 0
       v_t  + u v_x  + v v_y  = 0

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

    -------
    Returns
    -------
    out : ndarray, ndarray
        The u,v fluxes on the x- and y-interfaces

    """

    # Get the left and right interface states

    u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr = get_interface_states(grid, dt,
                                                                          u, v,
                                                                          ldelta_ux, ldelta_vx,
                                                                          ldelta_uy, ldelta_vy)

    # Solve for riemann problem for the second time

    # Get corrected normal advection velocity (MAC)

    u_MAC = incomp_interface.riemann(grid, u_xl, u_xr)
    v_MAC = incomp_interface.riemann(grid, v_yl, v_yr)

    # Upwind using the transverse corrected normal advective velocity

    ux = incomp_interface.upwind(grid, u_xl, u_xr, u_MAC)
    vx = incomp_interface.upwind(grid, v_xl, v_xr, u_MAC)

    uy = incomp_interface.upwind(grid, u_yl, u_yr, v_MAC)
    vy = incomp_interface.upwind(grid, v_yl, v_yr, v_MAC)

    # construct the flux

    fu_x = grid.scratch_array()
    fv_x = grid.scratch_array()
    fu_y = grid.scratch_array()
    fv_y = grid.scratch_array()

    fu_x.v(buf=1)[:, :] = 0.5 * ux.v(buf=1) * u_MAC.v(buf=1)
    fv_x.v(buf=1)[:, :] = 0.5 * vx.v(buf=1) * u_MAC.v(buf=1)

    fu_y.v(buf=1)[:, :] = 0.5 * uy.v(buf=1) * v_MAC.v(buf=1)
    fv_y.v(buf=1)[:, :] = 0.5 * vy.v(buf=1) * v_MAC.v(buf=1)

    return fu_x, fu_y, fv_x, fv_y
