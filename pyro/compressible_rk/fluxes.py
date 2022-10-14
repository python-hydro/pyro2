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

import pyro.compressible as comp
import pyro.compressible.interface as interface
import pyro.mesh.array_indexer as ai
import pyro.mesh.reconstruction as reconstruction
from pyro.util import msg


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

    gamma = rp.get_param("eos.gamma")

    # =========================================================================
    # compute the primitive variables
    # =========================================================================
    # Q = (rho, u, v, p)

    q = comp.cons_to_prim(my_data.data, gamma, ivars, myg)

    # =========================================================================
    # compute the flattening coefficients
    # =========================================================================

    # there is a single flattening coefficient (xi) for all directions
    use_flattening = rp.get_param("compressible.use_flattening")

    if use_flattening:
        xi_x = reconstruction.flatten(myg, q, 1, ivars, rp)
        xi_y = reconstruction.flatten(myg, q, 2, ivars, rp)

        xi = reconstruction.flatten_multid(myg, q, xi_x, xi_y, ivars)
    else:
        xi = 1.0

    # monotonized central differences in x-direction
    tm_limit = tc.timer("limiting")
    tm_limit.begin()

    limiter = rp.get_param("compressible.limiter")

    ldx = myg.scratch_array(nvar=ivars.nvar)
    ldy = myg.scratch_array(nvar=ivars.nvar)

    for n in range(ivars.nvar):
        ldx[:, :, n] = xi * reconstruction.limit(q[:, :, n], myg, 1, limiter)
        ldy[:, :, n] = xi * reconstruction.limit(q[:, :, n], myg, 2, limiter)

    tm_limit.end()

    # if we are doing a well-balanced scheme, then redo the pressure
    # note: we only have gravity in the y direction, so we will only
    # modify the y pressure slope
    well_balanced = rp.get_param("compressible.well_balanced")
    grav = rp.get_param("compressible.grav")

    if well_balanced:
        ldy[:, :, ivars.ip] = reconstruction.well_balance(
            q, myg, limiter, ivars, grav)

    # =========================================================================
    # x-direction
    # =========================================================================

    # left and right primitive variable states
    tm_states = tc.timer("interfaceStates")
    tm_states.begin()

    V_l = myg.scratch_array(ivars.nvar)
    V_r = myg.scratch_array(ivars.nvar)

    for n in range(ivars.nvar):
        V_l.ip(1, n=n, buf=2)[:, :] = q.v(n=n, buf=2) + 0.5 * ldx.v(n=n, buf=2)
        V_r.v(n=n, buf=2)[:, :] = q.v(n=n, buf=2) - 0.5 * ldx.v(n=n, buf=2)

    tm_states.end()

    # transform interface states back into conserved variables
    U_xl = comp.prim_to_cons(V_l, gamma, ivars, myg)
    U_xr = comp.prim_to_cons(V_r, gamma, ivars, myg)

    # =========================================================================
    # y-direction
    # =========================================================================

    # left and right primitive variable states
    tm_states.begin()

    for n in range(ivars.nvar):
        if well_balanced and n == ivars.ip:
            # we want to do p0 + p1 on the interfaces.  We found the
            # limited slope for p1 (it's average is 0).  So now we
            # need p0 on the interface too
            V_l.jp(1, n=n, buf=2)[:, :] = q.v(n=ivars.ip, buf=2) + \
                0.5 * myg.dy * q.v(n=ivars.irho, buf=2) * \
                grav + 0.5 * ldy.v(n=ivars.ip, buf=2)
            V_r.v(n=n, buf=2)[:, :] = q.v(n=ivars.ip, buf=2) - \
                0.5 * myg.dy * q.v(n=ivars.irho, buf=2) * \
                grav - 0.5 * ldy.v(n=ivars.ip, buf=2)
        else:
            V_l.jp(1, n=n, buf=2)[:, :] = q.v(
                n=n, buf=2) + 0.5 * ldy.v(n=n, buf=2)
            V_r.v(n=n, buf=2)[:, :] = q.v(n=n, buf=2) - 0.5 * ldy.v(n=n, buf=2)

    tm_states.end()

    # transform interface states back into conserved variables
    U_yl = comp.prim_to_cons(V_l, gamma, ivars, myg)
    U_yr = comp.prim_to_cons(V_r, gamma, ivars, myg)

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
    # apply artificial viscosity
    # =========================================================================
    cvisc = rp.get_param("compressible.cvisc")

    _ax, _ay = interface.artificial_viscosity(myg.ng, myg.dx, myg.dy,
        cvisc, q.v(n=ivars.iu, buf=myg.ng), q.v(n=ivars.iv, buf=myg.ng))

    avisco_x = ai.ArrayIndexer(d=_ax, grid=myg)
    avisco_y = ai.ArrayIndexer(d=_ay, grid=myg)

    b = (2, 1)

    for n in range(ivars.nvar):
        # F_x = F_x + avisco_x * (U(i-1,j) - U(i,j))
        var = my_data.get_var_by_index(n)

        F_x.v(buf=b, n=n)[:, :] += \
            avisco_x.v(buf=b) * (var.ip(-1, buf=b) - var.v(buf=b))

        # F_y = F_y + avisco_y * (U(i,j-1) - U(i,j))
        F_y.v(buf=b, n=n)[:, :] += \
            avisco_y.v(buf=b) * (var.jp(-1, buf=b) - var.v(buf=b))

    tm_flux.end()

    return F_x, F_y
