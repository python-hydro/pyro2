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
    u_MAC = burgers_interface.riemann_and_upwind(grid, u_xl, u_xr)
    v_MAC = burgers_interface.riemann_and_upwind(grid, v_yl, v_yr)

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
    u_xint = burgers_interface.upwind(grid, u_xl, u_xr, u_MAC)
    v_xint = burgers_interface.upwind(grid, v_xl, v_xr, u_MAC)
    u_yint = burgers_interface.upwind(grid, u_yl, u_yr, v_MAC)
    v_yint = burgers_interface.upwind(grid, v_yl, v_yr, v_MAC)

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
    uhat_adv = burgers_interface.riemann(grid, u_xl, u_xr)
    vhat_adv = burgers_interface.riemann(grid, v_yl, v_yr)

    # now that we have the advective velocities, upwind the left and right
    # states using the appropriate advective velocity.

    # on the x-interfaces, we upwind based on uhat_adv
    u_xint = burgers_interface.upwind(grid, u_xl, u_xr, uhat_adv)
    v_xint = burgers_interface.upwind(grid, v_xl, v_xr, uhat_adv)

    # on the y-interfaces, we upwind based on vhat_adv
    u_yint = burgers_interface.upwind(grid, u_yl, u_yr, vhat_adv)
    v_yint = burgers_interface.upwind(grid, v_yl, v_yr, vhat_adv)

    # at this point, these states are the `hat' states -- they only
    # considered the normal to the interface portion of the predictor.

    ubar = grid.scratch_array()
    vbar = grid.scratch_array()

    ubar.v(buf=2)[:, :] = 0.5 * (uhat_adv.v(buf=2) + uhat_adv.ip(1, buf=2))
    vbar.v(buf=2)[:, :] = 0.5 * (vhat_adv.v(buf=2) + vhat_adv.jp(1, buf=2))

    # v du/dy is the transerse term for the u states on x-interfaces
    vu_y = grid.scratch_array()
    vu_y.v(buf=2)[:, :] = vbar.v(buf=2) * (u_yint.jp(1, buf=2) - u_yint.v(buf=2))

    u_xl.ip(1, buf=2)[:, :] += -0.5 * dtdy * vu_y.v(buf=2) - 0.5 * dt * gradp_x.v(buf=2)
    u_xr.v(buf=2)[:, :] += -0.5 * dtdy * vu_y.v(buf=2) - 0.5 * dt * gradp_x.v(buf=2)

    # v dv/dy is the transverse term for the v states on x-interfaces
    vv_y = grid.scratch_array()
    vv_y.v(buf=2)[:, :] = vbar.v(buf=2) * (v_yint.jp(1, buf=2) - v_yint.v(buf=2))

    v_xl.ip(1, buf=2)[:, :] += -0.5 * dtdy * vv_y.v(buf=2) - 0.5 * dt * gradp_y.v(buf=2)
    v_xr.v(buf=2)[:, :] += -0.5 * dtdy * vv_y.v(buf=2) - 0.5 * dt * gradp_y.v(buf=2)

    # u dv/dx is the transverse term for the v states on y-interfaces
    uv_x = grid.scratch_array()
    uv_x.v(buf=2)[:, :] = ubar.v(buf=2) * (v_xint.ip(1, buf=2) - v_xint.v(buf=2))

    v_yl.jp(1, buf=2)[:, :] += -0.5 * dtdx * uv_x.v(buf=2) - 0.5 * dt * gradp_y.v(buf=2)
    v_yr.v(buf=2)[:, :] += -0.5 * dtdx * uv_x.v(buf=2) - 0.5 * dt * gradp_y.v(buf=2)

    # u du/dx is the transverse term for the u states on y-interfaces
    uu_x = grid.scratch_array()
    uu_x.v(buf=2)[:, :] = ubar.v(buf=2) * (u_xint.ip(1, buf=2) - u_xint.v(buf=2))

    u_yl.jp(1, buf=2)[:, :] += -0.5 * dtdx * uu_x.v(buf=2) - 0.5 * dt * gradp_x.v(buf=2)
    u_yr.v(buf=2)[:, :] += -0.5 * dtdx * uu_x.v(buf=2) - 0.5 * dt * gradp_x.v(buf=2)

    return u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr
