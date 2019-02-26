"""
This is a 2nd-order PLM method for a method-of-lines integration
(i.e., no characteristic tracing).

We wish to solve

.. math::

   U_t + F^x_x + F^y_y = H

we want U_{i+1/2} -- the interface values that are input to
the Riemann problem through the faces for each zone.

Taylor expanding *in space only* yields::

                              dU
   U          = U   + 0.5 dx  --
    i+1/2,j,L    i,j          dx

"""

import numpy as np

import compressible.interface as interface
import compressible as comp
import mesh.reconstruction as reconstruction
import mesh.array_indexer as ai

from util import msg


def fluxes(my_data, rp, ivars, solid, tc):
    """
    unsplitFluxes returns the fluxes through the x and y interfaces by
    doing an unsplit reconstruction of the interface values and then
    solving the Riemann problem through all the interfaces at once

    currently we assume a gamma-law EOS

    Parameters
    ----------
    my_data : CellCenterData2d object
        The data object containing the grid and advective scalar that
        we are advecting.
    rp : RuntimeParameters object
        The runtime parameters for the simulation
    vars : Variables object
        The Variables object that tells us which indices refer to which
        variables
    tc : TimerCollection object
        The timers we are using to profile

    Returns
    -------
    out : ndarray, ndarray
        The fluxes on the x- and y-interfaces

    """

    tm_flux = tc.timer("unsplitFluxes")
    tm_flux.begin()

    myg = my_data.grid
    U = my_data.data

    # =========================================================================
    # transform by rotating the velocity components
    # =========================================================================

    U_xl = myg.scratch_array(nvar=ivars.nvar)
    U_xr = myg.scratch_array(nvar=ivars.nvar)

    U_yl = myg.scratch_array(nvar=ivars.nvar)
    U_yr = myg.scratch_array(nvar=ivars.nvar)

    b = 3

    for n in range(ivars.nvar):
        U_xl.v(n=n, buf=b)[:, :] = myg.R_fcy.v(buf=b) * U.ip(-1, n=n, buf=b)
        U_xr.v(n=n, buf=b)[:, :] = myg.R_fcy.v(buf=b) * U.v(n=n, buf=b)
        U_yl.v(n=n, buf=b)[:, :] = myg.R_fcx.v(buf=b) * U.jp(-1, n=n, buf=b)
        U_yr.v(n=n, buf=b)[:, :] = myg.R_fcx.v(buf=b) * U.v(n=n, buf=b)

    gamma = rp.get_param("eos.gamma")
    #
    # # =========================================================================
    # # compute the primitive variables
    # # =========================================================================
    # # Q = (rho, u, v, p)
    #
    # q = comp.cons_to_prim(my_data.data, gamma, ivars, myg)
    #
    # # =========================================================================
    # # compute the flattening coefficients
    # # =========================================================================
    #
    # # there is a single flattening coefficient (xi) for all directions
    # use_flattening = rp.get_param("compressible.use_flattening")
    #
    # if use_flattening:
    #     xi_x = reconstruction.flatten(myg, q, 1, ivars, rp)
    #     xi_y = reconstruction.flatten(myg, q, 2, ivars, rp)
    #
    #     xi = reconstruction.flatten_multid(myg, q, xi_x, xi_y, ivars)
    # else:
    #     xi = 1.0
    #
    # # monotonized central differences in x-direction
    # tm_limit = tc.timer("limiting")
    # tm_limit.begin()
    #
    # limiter = rp.get_param("compressible.limiter")
    #
    # ldx = myg.scratch_array(nvar=ivars.nvar)
    # ldy = myg.scratch_array(nvar=ivars.nvar)
    #
    # for n in range(ivars.nvar):
    #     ldx[:, :, n] = xi * reconstruction.limit(q[:, :, n], myg, 1, limiter)
    #     ldy[:, :, n] = xi * reconstruction.limit(q[:, :, n], myg, 2, limiter)
    #
    # tm_limit.end()
    #
    # # =========================================================================
    # # x-direction
    # # =========================================================================
    #
    # # left and right primitive variable states
    # tm_states = tc.timer("interfaceStates")
    # tm_states.begin()
    #
    # V_l = myg.scratch_array(ivars.nvar)
    # V_r = myg.scratch_array(ivars.nvar)
    #
    # for n in range(ivars.nvar):
    #     V_l.ip(1, n=n, buf=2)[:, :] = q.v(n=n, buf=2) + 0.5 * ldx.v(n=n, buf=2)
    #     V_r.v(n=n, buf=2)[:, :] = q.v(n=n, buf=2) - 0.5 * ldx.v(n=n, buf=2)
    #
    # tm_states.end()
    #
    # # transform interface states back into conserved variables
    # U_xl = comp.prim_to_cons(V_l, gamma, ivars, myg)
    # U_xr = comp.prim_to_cons(V_r, gamma, ivars, myg)
    #
    # # =========================================================================
    # # y-direction
    # # =========================================================================
    #
    # # left and right primitive variable states
    # tm_states.begin()
    #
    # for n in range(ivars.nvar):
    #     V_l.jp(1, n=n, buf=2)[:, :] = q.v(n=n, buf=2) + 0.5 * ldy.v(n=n, buf=2)
    #     V_r.v(n=n, buf=2)[:, :] = q.v(n=n, buf=2) - 0.5 * ldy.v(n=n, buf=2)
    #
    # tm_states.end()
    #
    # # transform interface states back into conserved variables
    # U_yl = comp.prim_to_cons(V_l, gamma, ivars, myg)
    # U_yr = comp.prim_to_cons(V_r, gamma, ivars, myg)

    # =========================================================================
    # construct the fluxes normal to the interfaces
    # =========================================================================
    tm_riem = tc.timer("Riemann")
    tm_riem.begin()

    riemann = rp.get_param("compressible.riemann")

    if riemann == "HLLC":
        riemannFunc = interface.riemann_hllc
    elif riemann == "CGF":
        riemannFunc = interface.riemann_cgf
    else:
        msg.fail("ERROR: Riemann solver undefined")

    _fx = riemannFunc(1, myg.ng,
                      ivars.idens, ivars.ixmom, ivars.iymom, ivars.iener, ivars.irhox, ivars.naux,
                      solid.xl, solid.xr,
                      gamma, U_xl, U_xr)

    _fy = riemannFunc(2, myg.ng,
                      ivars.idens, ivars.ixmom, ivars.iymom, ivars.iener, ivars.irhox, ivars.naux,
                      solid.yl, solid.yr,
                      gamma, U_yl, U_yr)

    F_x = ai.ArrayIndexer(d=_fx, grid=myg)
    F_y = ai.ArrayIndexer(d=_fy, grid=myg)

    tm_riem.end()

    # =========================================================================
    # transform back
    # =========================================================================

    nx = myg.qx - 2 * myg.ng
    ny = myg.qy - 2 * myg.ng
    ilo = myg.ng
    ihi = myg.ng + nx
    jlo = myg.ng
    jhi = myg.ng + ny

    A_plus_delta_q_x = myg.scratch_array(nvar=ivars.nvar)
    A_minus_delta_q_x = myg.scratch_array(nvar=ivars.nvar)
    A_plus_delta_q_y = myg.scratch_array(nvar=ivars.nvar)
    A_minus_delta_q_y = myg.scratch_array(nvar=ivars.nvar)

    for i in range(ilo - 1, ihi + 1):
        for j in range(jlo - 1, jhi + 1):

            A_plus_delta_q_x[i, j, :] = myg.gamma_fcy[i,j] * myg.R_T_fcy[i,j] * (interface.consFlux(1, gamma, ivars.idens,
                                                            ivars.ixmom, ivars.iymom,
                                                            ivars.iener, ivars.irhox,
                                                            ivars.naux, U_xr[i, j, :])
                                         - F_x[i, j, :])
            A_minus_delta_q_x[i, j, :] = myg.gamma_fcy[i,j] * myg.R_T_fcy[i,j] * (F_x[i, j, :] -
                                          interface.consFlux(1, gamma, ivars.idens,
                                                             ivars.ixmom, ivars.iymom, ivars.iener,
                                                             ivars.irhox, ivars.naux,
                                                             U_xl[i, j, :]))

            A_plus_delta_q_y[i, j, :] = myg.gamma_fcx[i,j] * myg.R_T_fcx[i,j] * (interface.consFlux(2, gamma, ivars.idens,
                                                            ivars.ixmom, ivars.iymom,
                                                            ivars.iener, ivars.irhox,
                                                            ivars.naux, U_yr[i, j, :])
                                         - F_y[i, j, :])
            A_minus_delta_q_y[i, j, :] = myg.gamma_fcx[i,j] * myg.R_T_fcx[i,j] * (F_y[i, j, :] -
                                          interface.consFlux(2, gamma, ivars.idens,
                                                             ivars.ixmom, ivars.iymom,
                                                             ivars.iener, ivars.irhox,
                                                             ivars.naux, U_yl[i, j, :]))

    # q_star_x = U_xl[:, :, :] + F_x[:, :, :]
    #
    # A_plus_delta_q_x = myg.gamma_fcy * myg.R_T_fcy * \
    #     (interface.consFlux(U_xr) - interface.consFlux(q_star_x))
    # A_minus_delta_q_x = myg.gamma_fcy * myg.R_T_fcy * \
    #     (interface.consFlux(q_star_x) - interface.consFlux(U_xl))
    #
    # q_star_y = U_yl[:, :, :] + F_y[:, :, :]
    #
    # A_plus_delta_q_y = myg.gamma_fcx * myg.R_T_fcx * \
    #     (interface.consFlux(U_yr) - interface.consFlux(q_star_y))
    # A_minus_delta_q_y = myg.gamma_fcx * myg.R_T_fcx * \
    #     (interface.consFlux(q_star_y) - interface.consFlux(U_yl))

    tm_flux.end()

    return A_plus_delta_q_x, A_minus_delta_q_x, A_plus_delta_q_y, A_minus_delta_q_y
