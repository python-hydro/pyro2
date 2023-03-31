import numpy as np

import pyro.mesh.reconstruction as reconstruction
import pyro.incompressible.incomp_interface as interface


def unsplit_fluxes(my_data, rp, dt):
    r"""
    Construct the interface states for the burgers equation:

    .. math::

       u_t  + u u_x  + v u_y  = 0
       v_t  + u v_x  + v v_y  = 0

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
    my_data : CellCenterData2d object
        The data object containing the grid and advective scalar that
        we are advecting.
    rp : RuntimeParameters object
        The runtime parameters for the simulation
    dt : float
        The timestep we are advancing through.

    Returns
    -------
    out : ndarray, ndarray
        The u,v fluxes on the x- and y-interfaces

    """

    myg = my_data.grid

    u = my_data.get_var("x-velocity")
    v = my_data.get_var("y-velocity")

    # cx = myg.scratch_array()
    # cy = myg.scratch_array()

    # cx.v(buf=1)[:, :] = u.v(buf=1)*dt/myg.dx
    # cy.v(buf=1)[:, :] = v.v(buf=1)*dt/myg.dy

    dtdx = dt/myg.dx
    dtdy = dt/myg.dy
    
    # --------------------------------------------------------------------------
    # monotonized central differences
    # --------------------------------------------------------------------------

    limiter = rp.get_param("advection.limiter")

    # Give da/dx and da/dy using input: (state, grid, direction, limiter)

    ldelta_ux = reconstruction.limit(u, myg, 1, limiter)
    ldelta_uy = reconstruction.limit(u, myg, 2, limiter)
    ldelta_vx = reconstruction.limit(v, myg, 1, limiter)
    ldelta_vy = reconstruction.limit(v, myg, 2, limiter)

    ul_x = myg.scratch_array()
    ur_x = myg.scratch_array()
    ul_y = myg.scratch_array()
    ur_y = myg.scratch_array()

    vl_x = myg.scratch_array()
    vr_x = myg.scratch_array()
    vl_y = myg.scratch_array()
    vr_y = myg.scratch_array()
    
    # Determine left and right interface states in x and y.

    # First compute the predictor terms for both u and v states.

    ul_x.v(buf=1)[:, :] = u.ip(-1, buf=1) + 0.5*(1.0 - dtdx*u.ip(-1, buf=1))*ldelta_ux.ip(-1, buf=1)
    ul_y.v(buf=1)[:, :] = u.jp(-1, buf=1) + 0.5*(1.0 - dtdy*v.jp(-1, buf=1))*ldelta_uy.jp(-1, buf=1)
    ur_x.v(buf=1)[:, :] = u.v(buf=1) - 0.5*(1.0 + dtdx*u.v(buf=1))*ldelta_ux.v(buf=1)
    ur_y.v(buf=1)[:, :] = u.v(buf=1) - 0.5*(1.0 + dtdy*v.v(buf=1))*ldelta_uy.v(buf=1)

    vl_x.v(buf=1)[:, :] = v.ip(-1, buf=1) + 0.5*(1.0 - dtdx*u.ip(-1, buf=1))*ldelta_vx.ip(-1, buf=1)
    vl_y.v(buf=1)[:, :] = v.jp(-1, buf=1) + 0.5*(1.0 - dtdy*v.jp(-1, buf=1))*ldelta_vy.jp(-1, buf=1)
    vr_x.v(buf=1)[:, :] = v.v(buf=1) - 0.5*(1.0 + dtdx*u.v(buf=1))*ldelta_vx.v(buf=1)
    vr_y.v(buf=1)[:, :] = v.v(buf=1) - 0.5*(1.0 + dtdy*v.v(buf=1))*ldelta_vy.v(buf=1)
    
    # Solve Riemann's problem to get the correct transverse term

    # first get the normal advective velocity through each x and y interface

    uhat_adv = interface.riemann(myg, ul_x, ur_x)
    vhat_adv = interface.riemann(myg, vl_y, vr_y)

    # Upwind the l and r states using the normal advective velocity through x and y interface

    ux_int = interface.upwind(myg, ul_x, ur_x, uhat_adv)
    vx_int = interface.upwind(myg, vl_x, vr_x, uhat_adv)

    uy_int = interface.upwind(myg, ul_y, ur_y, vhat_adv)
    vy_int = interface.upwind(myg, vl_y, vr_y, vhat_adv)

    # Calculate the average normal interface velocities

    ubar_adv = myg.scratch_array()
    vbar_adv = myg.scratch_array()

    ubar_adv.v(buf=1)[:, :] = 0.5*(uhat_adv.v(buf=1) + uhat_adv.ip(-1, buf=1))
    vbar_adv.v(buf=1)[:, :] = 0.5*(vhat_adv.v(buf=1) + vhat_adv.jp(-1, buf=1))

    # Apply transverse correction terms:

    ul_x.v(buf=1)[:, :] -= 0.5 * dtdy * vbar_adv.v(buf=1) * (uy_int.ip_jp(-1, 1, buf=1) - uy_int.ip_jp(-1, 0, buf=1))
    ul_y.v(buf=1)[:, :] -= 0.5 * dtdx * ubar_adv.v(buf=1) * (ux_int.ip_jp(1, -1, buf=1) - ux_int.ip_jp(0, -1, buf=1))
    ur_x.v(buf=1)[:, :] -= 0.5 * dtdy * vbar_adv.v(buf=1) * (uy_int.ip_jp(0, 1, buf=1) - uy_int.ip_jp(0, 0, buf=1))
    ur_y.v(buf=1)[:, :] -= 0.5 * dtdx * ubar_adv.v(buf=1) * (ux_int.ip_jp(1, 0, buf=1) - ux_int.ip_jp(0, 0, buf=1))

    vl_x.v(buf=1)[:, :] -= 0.5 * dtdy * vbar_adv.v(buf=1) * (vy_int.ip_jp(-1, 1, buf=1) - vy_int.ip_jp(-1, 0, buf=1))
    vl_y.v(buf=1)[:, :] -= 0.5 * dtdx * ubar_adv.v(buf=1) * (vx_int.ip_jp(1, -1, buf=1) - vx_int.ip_jp(0, -1, buf=1))
    vr_x.v(buf=1)[:, :] -= 0.5 * dtdy * vbar_adv.v(buf=1) * (vy_int.ip_jp(0, 1, buf=1) - vy_int.ip_jp(0, 0, buf=1))
    vr_y.v(buf=1)[:, :] -= 0.5 * dtdx * ubar_adv.v(buf=1) * (vx_int.ip_jp(1, 0, buf=1) - vx_int.ip_jp(0, 0, buf=1))

    # Solve for riemann problem for the second time
    
    # Get corrected normal advection velocity (MAC)

    u_MAC = interface.riemann(myg, ul_x, ur_x)
    v_MAC = interface.riemann(myg, vl_y, vr_y)

    # Upwind using the transverse corrected normal advective velocity

    ux = interface.upwind(myg, ul_x, ur_x, u_MAC)
    vx = interface.upwind(myg, vl_x, vr_x, u_MAC)

    uy = interface.upwind(myg, ul_y, ur_y, v_MAC)
    vy = interface.upwind(myg, vl_y, vr_y, v_MAC)

    # construct the flux

    fu_x = myg.scratch_array()
    fv_x = myg.scratch_array()
    fu_y = myg.scratch_array()
    fv_y = myg.scratch_array()

    fu_x.v(buf=1)[:, :] = 0.5 * ux.v(buf=1) * ux.v(buf=1)
    fv_x.v(buf=1)[:, :] = 0.5 * vx.v(buf=1) * ux.v(buf=1)

    fu_y.v(buf=1)[:, :] = 0.5 * vy.v(buf=1) * uy.v(buf=1)
    fv_y.v(buf=1)[:, :] = 0.5 * vy.v(buf=1) * vy.v(buf=1)

    return fu_x, fu_y, fv_x, fv_y
