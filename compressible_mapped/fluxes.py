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

    We use the method of Calhoun, D. A., Helzel, C., & LeVeque, R. J. (2008). SIAM review, 50(4), 723-752 for the mapped grids.

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
    gamma = rp.get_param("eos.gamma")

    # =========================================================================
    # transform by rotating the velocity components
    # =========================================================================

    U_xl = myg.scratch_array(nvar=ivars.nvar)
    U_xr = myg.scratch_array(nvar=ivars.nvar)

    U_yl = myg.scratch_array(nvar=ivars.nvar)
    U_yr = myg.scratch_array(nvar=ivars.nvar)

    b = 3

    nx, ny, _ = np.shape(U_xl.v(buf=b))

    for i in range(nx):
        for j in range(ny):
            U_xl.v(buf=b)[i, j] = (my_data.R_fcx.v(buf=b)[i, j] @
                                   U.ip(-1, buf=b)[i, j])
            U_xr.v(buf=b)[i, j] = (my_data.R_fcx.v(buf=b)[i, j] @
                                   U.v(buf=b)[i, j])
            U_yl.v(buf=b)[i, j] = (my_data.R_fcy.v(buf=b)[i, j] @
                                   U.jp(-1, buf=b)[i, j])
            U_yr.v(buf=b)[i, j] = (my_data.R_fcy.v(buf=b)[i, j] @
                                   U.v(buf=b)[i, j])

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

            A_plus_delta_q_x[i, j, :] = myg.gamma_fcx[i, j] * \
                (my_data.R_fcx[i, j]).T  @ \
                (interface.consFlux(1, gamma, ivars.idens, ivars.ixmom, ivars.iymom,
                                    ivars.iener, ivars.irhox, ivars.naux,
                                    U_xr[i, j, :]) - F_x[i, j, :])
            A_minus_delta_q_x[i, j, :] = myg.gamma_fcx[i, j] * \
                (my_data.R_fcx[i, j]).T @ \
                (F_x[i, j, :] - interface.consFlux(1, gamma, ivars.idens, ivars.ixmom,
                                                   ivars.iymom, ivars.iener,
                                                   ivars.irhox, ivars.naux,
                                                   U_xl[i, j, :]))

            A_plus_delta_q_y[i, j, :] = myg.gamma_fcy[i, j] * \
                (my_data.R_fcy[i, j]).T @ \
                (interface.consFlux(2, gamma, ivars.idens, ivars.ixmom, ivars.iymom,
                                    ivars.iener, ivars.irhox, ivars.naux,
                                    U_yr[i, j, :]) - F_y[i, j, :])
            A_minus_delta_q_y[i, j, :] = myg.gamma_fcy[i, j] * \
                (my_data.R_fcy[i, j]).T @ \
                (F_y[i, j, :] - interface.consFlux(2, gamma, ivars.idens, ivars.ixmom,
                                                   ivars.iymom, ivars.iener,
                                                   ivars.irhox, ivars.naux,
                                                   U_yl[i, j, :]))

    tm_flux.end()

    return A_plus_delta_q_x, A_minus_delta_q_x, A_plus_delta_q_y, A_minus_delta_q_y
