import numpy as np

from pyro.burgers import burgers_interface
from pyro.multigrid import MG


def get_lap(grid, a):
    r"""
    Parameters
    ----------
    grid: grid object
    a: ndarray
       the variable that we want to find the laplacian of

    Returns
    -------
    out : ndarray (laplacian of state a)
    """

    lap = grid.scratch_array()
    
    lap.v(buf=2)[:, :] = (a.ip(1, buf=2) - 2.0*a.v(buf=2) + a.ip(-1, buf=2)) / grid.dx**2 \
                       + (a.jp(1, buf=2) - 2.0*a.v(buf=2) + a.jp(-1, buf=2)) / grid.dy**2

    return lap

def diffuse(my_data, rp, dt, scalar_name, A):
    r"""
    A routine to solve the Helmhotlz equation with constant coefficient
    and update the state.

    (a + b \lap) phi = f

    using Crank-Nicolson discretization with multigrid V-cycle.
    
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
    A: ndarray
       The advective source term for diffusion

    Returns
    -------
    out : ndarray (solution of the Helmholtz equation)

    """

    myg = my_data.grid

    a = my_data.get_var(scalar_name)
    eps = rp.get_param("diffusion.eps")

    # Create the multigrid with a constant diffusion coefficient
    # the equation has form:
    # (alpha - beta L) phi = f
    #
    # with alpha = 1
    #      beta  = (dt/2) k
    #      f     = phi + (dt/2) k L phi - A
    #
    # Same as Crank-Nicolson discretization except with an extra
    # advection source term)

    mg = MG.CellCenterMG2d(myg.nx, myg.ny,
                           xmin=myg.xmin, xmax=myg.xmax,
                           ymin=myg.ymin, ymax=myg.ymax,
                           xl_BC_type=my_data.BCs[scalar_name].xlb,
                           xr_BC_type=my_data.BCs[scalar_name].xrb,
                           yl_BC_type=my_data.BCs[scalar_name].ylb,
                           yr_BC_type=my_data.BCs[scalar_name].yrb,
                           alpha=1.0, beta=0.5*dt*eps,
                           verbose=0)

    # Compute the RHS: f

    f = mg.soln_grid.scratch_array()
    
    lap = get_lap(myg, a)
    
    f.v()[:, :] = a.v() + 0.5*dt*eps * lap.v() - dt*A.v()

    mg.init_RHS(f)

    # initial guess is zeros
    
    mg.init_zeros()
    
    mg.solve(rtol=1.e-12)

    # perform the diffusion update

    a.v()[:, :] = mg.get_solution().v()

def apply_diffusion_corrections(grid, dt, eps,
                                u, v,
                                u_xl, u_xr,
                                u_yl, u_yr,
                                v_xl, v_xr,
                                v_yl, v_yr):
    r"""
    Apply diffusion correction term to the interface state

    .. math::

       u_t  + u u_x  + v u_y  = eps (u_xx + u_yy)
       v_t  + u v_x  + v v_y  = eps (v_xx + v_yy)

    We use a second-order (piecewise linear) unsplit Godunov method
    (following Colella 1990).

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
    grid : Grid2d
        The grid object
    dt : float
        The timestep
    eps : float
         The viscosity
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
        unsplit predictions of the left and right states of u and v on both the x- and
        y-interfaces along with diffusion correction terms.
    """

    # Get the interface states from pure advection

    # u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr = burgers_interface.get_interface_states(grid, dt,
    #                      u, v,
    #                      ldelta_ux, ldelta_vx,
    #                      ldelta_uy, ldelta_vy)

    #apply diffusion correction to the interface

    # ud = grid.scratch_array()
    # vd = grid.scratch_array()

    lap_u = get_lap(grid, u)
    lap_v = get_lap(grid, v)

    # ud.v(buf=2)[:, :] = 0.5 * eps * dt * lap_u.v(buf=2)
    # vd.v(buf=2)[:, :] = 0.5 * eps * dt * lap_v.v(buf=2)

    u_xl.ip(1, buf=2)[:, :] += 0.5 * eps * dt * lap_u.v(buf=2)
    u_yl.jp(1, buf=2)[:, :] += 0.5 * eps * dt * lap_u.v(buf=2)
    u_xr.v(buf=2)[:, :] += 0.5 * eps * dt * lap_u.v(buf=2)
    u_yr.v(buf=2)[:, :] += 0.5 * eps * dt * lap_u.v(buf=2)

    v_xl.ip(1, buf=2)[:, :] += 0.5 * eps * dt * lap_v.v(buf=2)
    v_yl.jp(1, buf=2)[:, :] += 0.5 * eps * dt * lap_v.v(buf=2)
    v_xr.v(buf=2)[:, :] += 0.5 * eps * dt * lap_v.v(buf=2)
    v_yr.v(buf=2)[:, :] += 0.5 * eps * dt * lap_v.v(buf=2)

    
    # # now get the normal advective velocities on the interfaces by solving
    # # the Riemann problem.
    # uhat_adv = burgers_interface.riemann(grid, u_xl, u_xr)
    # vhat_adv = burgers_interface.riemann(grid, v_yl, v_yr)

    # # now that we have the advective velocities, upwind the left and right
    # # states using the appropriate advective velocity.

    # # on the x-interfaces, we upwind based on uhat_adv
    # u_xint = burgers_interface.upwind(grid, u_xl, u_xr, uhat_adv)
    # v_xint = burgers_interface.upwind(grid, v_xl, v_xr, uhat_adv)

    # # on the y-interfaces, we upwind based on vhat_adv
    # u_yint = burgers_interface.upwind(grid, u_yl, u_yr, vhat_adv)
    # v_yint = burgers_interface.upwind(grid, v_yl, v_yr, vhat_adv)

    # # at this point, these states are the `hat' states -- they only
    # # considered the normal to the interface portion of the predictor.

    # ubar = grid.scratch_array()
    # vbar = grid.scratch_array()

    # ubar.v(buf=2)[:, :] = 0.5 * (uhat_adv.v(buf=2) + uhat_adv.ip(1, buf=2))
    # vbar.v(buf=2)[:, :] = 0.5 * (vhat_adv.v(buf=2) + vhat_adv.jp(1, buf=2))

    # # the transverse term for the u states on x-interfaces
    # u_xl.ip(1, buf=2)[:, :] += -0.5 * dtdy * vbar.v(buf=2) * (u_yint.jp(1, buf=2) - u_yint.v(buf=2))
    # u_xr.v(buf=2)[:, :] += -0.5 * dtdy * vbar.v(buf=2) * (u_yint.jp(1, buf=2) - u_yint.v(buf=2))

    # # the transverse term for the v states on x-interfaces
    # v_xl.ip(1, buf=2)[:, :] += -0.5 * dtdy * vbar.v(buf=2) * (v_yint.jp(1, buf=2) - v_yint.v(buf=2))
    # v_xr.v(buf=2)[:, :] += -0.5 * dtdy * vbar.v(buf=2) * (v_yint.jp(1, buf=2) - v_yint.v(buf=2))

    # # the transverse term for the v states on y-interfaces
    # v_yl.jp(1, buf=2)[:, :] += -0.5 * dtdx * ubar.v(buf=2) * (v_xint.ip(1, buf=2) - v_xint.v(buf=2))
    # v_yr.v(buf=2)[:, :] += -0.5 * dtdx * ubar.v(buf=2) * (v_xint.ip(1, buf=2) - v_xint.v(buf=2))

    # # the transverse term for the u states on y-interfaces
    # u_yl.jp(1, buf=2)[:, :] += -0.5 * dtdx * ubar.v(buf=2) * (u_xint.ip(1, buf=2) - u_xint.v(buf=2))
    # u_yr.v(buf=2)[:, :] += -0.5 * dtdx * ubar.v(buf=2) * (u_xint.ip(1, buf=2) - u_xint.v(buf=2))

    
    
    # # eps = rp.get_param("diffusion.eps")
    

    # # Solve for riemann problem for the second time

    # # Get corrected normal advection velocity

    # # u_diffuse = burgers_interface.riemann(grid, u_xl, u_xr)
    # # v_diffuse = burgers_interface.riemann(grid, v_yl, v_yr)

    # # # # Upwind using the transverse corrected normal advective velocity

    # ux_d = burgers_interface.upwind(grid, u_xl, u_xr, u_diffuse)
    # vx_d = burgers_interface.upwind(grid, v_xl, v_xr, u_diffuse)

    # uy_d = burgers_interface.upwind(grid, u_yl, u_yr, v_diffuse)
    # vy_d = burgers_interface.upwind(grid, v_yl, v_yr, v_diffuse)
    
    # # # Compute the diffusion term:


    # ud = grid.scratch_array()
    # vd = grid.scratch_array()

    # lap_u = get_lap(grid, u)
    # lap_v = get_lap(grid, v)

    # # lap_u = get_lap(grid, u_diffuse)
    # # lap_v = get_lap(grid, v_diffuse)
    
    # ud.v(buf=2)[:, :] = 0.5 * eps * dt * lap_u.v(buf=2)
    # vd.v(buf=2)[:, :] = 0.5 * eps * dt * lap_v.v(buf=2)

    # # ud_x = grid.scratch_array()
    # # ud_y = grid.scratch_array()
    # # vd_x = grid.scratch_array()
    # # vd_y = grid.scratch_array()

    # # lap_ux = get_lap(grid, ux_d)
    # # lap_uy = get_lap(grid, uy_d)
    # # lap_vx = get_lap(grid, vx_d)
    # # lap_vy = get_lap(grid, vy_d)

    # # ud_x.v(buf=2)[:, :] = 0.5 * eps * dt * lap_uy.v(buf=2)
    # # ud_y.v(buf=2)[:, :] = 0.5 * eps * dt * lap_ux.v(buf=2)
    # # vd_x.v(buf=2)[:, :] = 0.5 * eps * dt * lap_vy.v(buf=2)
    # # vd_y.v(buf=2)[:, :] = 0.5 * eps * dt * lap_vx.v(buf=2)

    # # # apply diffusion term:

    # # u_xl.ip(1, buf=2)[:, :] += ud_x.v(buf=2)
    # # u_yl.jp(1, buf=2)[:, :] += ud_y.v(buf=2)
    # # u_xr.v(buf=2)[:, :] += ud_x.v(buf=2)
    # # u_yr.v(buf=2)[:, :] += ud_y.v(buf=2)

    # # v_xl.ip(1, buf=2)[:, :] += vd_x.v(buf=2)
    # # v_yl.jp(1, buf=2)[:, :] += vd_y.v(buf=2)
    # # v_xr.v(buf=2)[:, :] += vd_x.v(buf=2)
    # # v_yr.v(buf=2)[:, :] += vd_y.v(buf=2)
    
    
    # u_xl.ip(1, buf=2)[:, :] += ud.v(buf=2)
    # u_yl.jp(1, buf=2)[:, :] += ud.v(buf=2)
    # u_xr.v(buf=2)[:, :] += ud.v(buf=2)
    # u_yr.v(buf=2)[:, :] += ud.v(buf=2)

    # v_xl.ip(1, buf=2)[:, :] += vd.v(buf=2)
    # v_yl.jp(1, buf=2)[:, :] += vd.v(buf=2)
    # v_xr.v(buf=2)[:, :] += vd.v(buf=2)
    # v_yr.v(buf=2)[:, :] += vd.v(buf=2)
    
    return u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr

# def unsplit_fluxes(grid, dt,
#                    u, v,
#                    ldelta_ux, ldelta_vx,
#                    ldelta_uy, ldelta_vy, eps):
#     r"""
#     Construct the interface fluxes for the viscous burgers equation:

#     Parameters
#     ----------
#     grid : Grid2d
#         The grid object
#     dt : float
#         The timestep
#     u, v : ndarray
#         x-velocity and y-velocity
#     ldelta_ux, ldelta_uy: ndarray
#         Limited slopes of the x-velocity in the x and y directions
#     ldelta_vx, ldelta_vy: ndarray
#         Limited slopes of the y-velocity in the x and y directions
#     eps: float
#          the viscosity
#     -------
#     Returns
#     -------
#     out : ndarray, ndarray
#         The u,v fluxes on the x- and y-interfaces

#     """

#     # Get the interface states without transverse or diffusion corrections
#     u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr = burgers_interface.get_interface_states(grid, dt,
#                                                                                             u, v,
#                                                                                             ldelta_ux, ldelta_vx,
#                                                                                             ldelta_uy, ldelta_vy)

#     # Apply diffusion correction terms to the interface states
#     u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr = apply_diffusion_corrections(grid, dt, eps,
#                                                                                  u, v,
#                                                                                  u_xl, u_xr,
#                                                                                  u_yl, u_yr,
#                                                                                  v_xl, v_xr,
#                                                                                  v_yl, v_yr)
#     # Apply transverse correction terms to the interface states
#     u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr = apply_transverse_corrections(grid, dt,
#                                                                                   u_xl, u_xr,
#                                                                                   u_yl, u_yr,
#                                                                                   v_xl, v_xr,
#                                                                                   v_yl, v_yr)
    
    
#     # construct the advective flux

#     u_MAC = burgers_interface.riemann_and_upwind(grid, u_xl, u_xr)
#     v_MAC = burgers_interface.riemann_and_upwind(grid, v_yl, v_yr)

#     # Upwind using the transverse corrected normal advective velocity

#     ux = burgers_interface.upwind(grid, u_xl, u_xr, u_MAC)
#     vx = burgers_interface.upwind(grid, v_xl, v_xr, u_MAC)

#     uy = burgers_interface.upwind(grid, u_yl, u_yr, v_MAC)
#     vy = burgers_interface.upwind(grid, v_yl, v_yr, v_MAC)
    
#     fu_x = grid.scratch_array()
#     fv_x = grid.scratch_array()
#     fu_y = grid.scratch_array()
#     fv_y = grid.scratch_array()

#     fu_x.v(buf=2)[:, :] = 0.5 * ux.v(buf=2) * u_MAC.v(buf=2)
#     fv_x.v(buf=2)[:, :] = 0.5 * vx.v(buf=2) * u_MAC.v(buf=2)

#     fu_y.v(buf=2)[:, :] = 0.5 * uy.v(buf=2) * v_MAC.v(buf=2)
#     fv_y.v(buf=2)[:, :] = 0.5 * vy.v(buf=2) * v_MAC.v(buf=2)

#     return fu_x, fu_y, fv_x, fv_y


