"""
Implementation of the Colella 2nd order unsplit Godunov scheme.  This
is a 2-dimensional implementation only.  We assume that the grid is
uniform, but it is relatively straightforward to relax this
assumption.

There are several different options for this solver (they are all
discussed in the Colella paper).

  limiter          = 0 to use no limiting
                   = 1 to use the 2nd order MC limiter
                   = 2 to use the 4th order MC limiter

  riemann          = HLLC to use the HLLC solver
                   = CGF to use the Colella, Glaz, and Ferguson solver

  use_flattening   = 1 to use the multidimensional flattening
                     algorithm at shocks

  delta, z0, z1      these are the flattening parameters.  The default
                     are the values listed in Colella 1990.

   j+3/2--+---------+---------+---------+
          |         |         |         |
     j+1 _|         |         |         |
          |         |         |         |
          |         |         |         |
   j+1/2--+---------XXXXXXXXXXX---------+
          |         X         X         |
       j _|         X         X         |
          |         X         X         |
          |         X         X         |
   j-1/2--+---------XXXXXXXXXXX---------+
          |         |         |         |
     j-1 _|         |         |         |
          |         |         |         |
          |         |         |         |
   j-3/2--+---------+---------+---------+
          |    |    |    |    |    |    |
              i-1        i        i+1
        i-3/2     i-1/2     i+1/2     i+3/2

We wish to solve

  U_t + F^x_x + F^y_y = H

we want U_{i+1/2}^{n+1/2} -- the interface values that are input to
the Riemann problem through the faces for each zone.

Taylor expanding yields in space only

                             dU
  U          = U   + 0.5 dx  --
   i+1/2,j,L    i,j          dx


Updating U_{i,j}:

  -- We want to find the state to the left and right (or top and
     bottom) of each interface, ex. U_{i+1/2,j,[lr]}^{n+1/2}, and use
     them to solve a Riemann problem across each of the four
     interfaces.
"""

import compressible.eos as eos
import compressible.interface_f as interface_f
import mesh.reconstruction as reconstruction
import mesh.reconstruction_f as reconstruction_f
import mesh.array_indexer as ai

from util import msg

def fluxes(my_data, rp, vars, solid, tc):
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

    #=========================================================================
    # compute the primitive variables
    #=========================================================================
    # Q = (rho, u, v, p)

    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    r, u, v, p = my_data.get_var("primitive")

    smallp = 1.e-10
    p = p.clip(smallp)   # apply a floor to the pressure


    #=========================================================================
    # compute the flattening coefficients
    #=========================================================================

    # there is a single flattening coefficient (xi) for all directions
    use_flattening = rp.get_param("compressible.use_flattening")

    if use_flattening:
        delta = rp.get_param("compressible.delta")
        z0 = rp.get_param("compressible.z0")
        z1 = rp.get_param("compressible.z1")

        xi_x = reconstruction_f.flatten(1, p, u, myg.qx, myg.qy, myg.ng, smallp, delta, z0, z1)
        xi_y = reconstruction_f.flatten(2, p, v, myg.qx, myg.qy, myg.ng, smallp, delta, z0, z1)

        xi = reconstruction_f.flatten_multid(xi_x, xi_y, p, myg.qx, myg.qy, myg.ng)
    else:
        xi = 1.0


    # monotonized central differences in x-direction
    tm_limit = tc.timer("limiting")
    tm_limit.begin()

    limiter = rp.get_param("compressible.limiter")

    ldelta_rx = xi*reconstruction.limit(r, myg, 1, limiter)
    ldelta_ux = xi*reconstruction.limit(u, myg, 1, limiter)
    ldelta_vx = xi*reconstruction.limit(v, myg, 1, limiter)
    ldelta_px = xi*reconstruction.limit(p, myg, 1, limiter)

    # monotonized central differences in y-direction
    ldelta_ry = xi*reconstruction.limit(r, myg, 2, limiter)
    ldelta_uy = xi*reconstruction.limit(u, myg, 2, limiter)
    ldelta_vy = xi*reconstruction.limit(v, myg, 2, limiter)
    ldelta_py = xi*reconstruction.limit(p, myg, 2, limiter)

    tm_limit.end()


    #=========================================================================
    # x-direction
    #=========================================================================

    # left and right primitive variable states
    tm_states = tc.timer("interfaceStates")
    tm_states.begin()

    V_l = myg.scratch_array(vars.nvar)
    V_r = myg.scratch_array(vars.nvar)

    V_l.ip(1, n=vars.irho, buf=2)[:,:] = r.v(buf=2) + 0.5*ldelta_rx.v(buf=2)
    V_r.v(n=vars.irho, buf=2)[:,:] = r.v(buf=2) - 0.5*ldelta_rx.v(buf=2)

    V_l.ip(1, n=vars.iu, buf=2)[:,:] = u.v(buf=2) + 0.5*ldelta_ux.v(buf=2)
    V_r.v(n=vars.iu, buf=2)[:,:] = u.v(buf=2) - 0.5*ldelta_ux.v(buf=2)

    V_l.ip(1, n=vars.iv, buf=2)[:,:] = v.v(buf=2) + 0.5*ldelta_vx.v(buf=2)
    V_r.v(n=vars.iv, buf=2)[:,:] = v.v(buf=2) - 0.5*ldelta_vx.v(buf=2)

    V_l.ip(1, n=vars.ip, buf=2)[:,:] = p.v(buf=2) + 0.5*ldelta_px.v(buf=2)
    V_r.v(n=vars.ip, buf=2)[:,:] = p.v(buf=2) - 0.5*ldelta_px.v(buf=2)

    tm_states.end()


    # transform interface states back into conserved variables
    U_xl = myg.scratch_array(vars.nvar)
    U_xr = myg.scratch_array(vars.nvar)

    U_xl[:,:,vars.idens] = V_l[:,:,vars.irho]
    U_xl[:,:,vars.ixmom] = V_l[:,:,vars.irho]*V_l[:,:,vars.iu]
    U_xl[:,:,vars.iymom] = V_l[:,:,vars.irho]*V_l[:,:,vars.iv]
    U_xl[:,:,vars.iener] = eos.rhoe(gamma, V_l[:,:,vars.ip]) + \
        0.5*V_l[:,:,vars.irho]*(V_l[:,:,vars.iu]**2 + V_l[:,:,vars.iv]**2)

    U_xr[:,:,vars.idens] = V_r[:,:,vars.irho]
    U_xr[:,:,vars.ixmom] = V_r[:,:,vars.irho]*V_r[:,:,vars.iu]
    U_xr[:,:,vars.iymom] = V_r[:,:,vars.irho]*V_r[:,:,vars.iv]
    U_xr[:,:,vars.iener] = eos.rhoe(gamma, V_r[:,:,vars.ip]) + \
        0.5*V_r[:,:,vars.irho]*(V_r[:,:,vars.iu]**2 + V_r[:,:,vars.iv]**2)



    #=========================================================================
    # y-direction
    #=========================================================================


    # left and right primitive variable states
    tm_states.begin()

    V_l.jp(1, n=vars.irho, buf=2)[:,:] = r.v(buf=2) + 0.5*ldelta_ry.v(buf=2)
    V_r.v(n=vars.irho, buf=2)[:,:] = r.v(buf=2) - 0.5*ldelta_ry.v(buf=2)

    V_l.jp(1, n=vars.iu, buf=2)[:,:] = u.v(buf=2) + 0.5*ldelta_uy.v(buf=2)
    V_r.v(n=vars.iu, buf=2)[:,:] = u.v(buf=2) - 0.5*ldelta_uy.v(buf=2)

    V_l.jp(1, n=vars.iv, buf=2)[:,:] = v.v(buf=2) + 0.5*ldelta_vy.v(buf=2)
    V_r.v(n=vars.iv, buf=2)[:,:] = v.v(buf=2) - 0.5*ldelta_vy.v(buf=2)

    V_l.jp(1, n=vars.ip, buf=2)[:,:] = p.v(buf=2) + 0.5*ldelta_py.v(buf=2)
    V_r.v(n=vars.ip, buf=2)[:,:] = p.v(buf=2) - 0.5*ldelta_py.v(buf=2)

    tm_states.end()


    # transform interface states back into conserved variables
    U_yl = myg.scratch_array(vars.nvar)
    U_yr = myg.scratch_array(vars.nvar)

    U_yl[:,:,vars.idens] = V_l[:,:,vars.irho]
    U_yl[:,:,vars.ixmom] = V_l[:,:,vars.irho]*V_l[:,:,vars.iu]
    U_yl[:,:,vars.iymom] = V_l[:,:,vars.irho]*V_l[:,:,vars.iv]
    U_yl[:,:,vars.iener] = eos.rhoe(gamma, V_l[:,:,vars.ip]) + \
        0.5*V_l[:,:,vars.irho]*(V_l[:,:,vars.iu]**2 + V_l[:,:,vars.iv]**2)

    U_yr[:,:,vars.idens] = V_r[:,:,vars.irho]
    U_yr[:,:,vars.ixmom] = V_r[:,:,vars.irho]*V_r[:,:,vars.iu]
    U_yr[:,:,vars.iymom] = V_r[:,:,vars.irho]*V_r[:,:,vars.iv]
    U_yr[:,:,vars.iener] = eos.rhoe(gamma, V_r[:,:,vars.ip]) + \
        0.5*V_r[:,:,vars.irho]*(V_r[:,:,vars.iu]**2 + V_r[:,:,vars.iv]**2)


    #=========================================================================
    # construct the fluxes normal to the interfaces
    #=========================================================================
    tm_riem = tc.timer("Riemann")
    tm_riem.begin()

    riemann = rp.get_param("compressible.riemann")

    if riemann == "HLLC":
        riemannFunc = interface_f.riemann_hllc
    elif riemann == "CGF":
        riemannFunc = interface_f.riemann_cgf
    else:
        msg.fail("ERROR: Riemann solver undefined")

    _fx = riemannFunc(1, myg.qx, myg.qy, myg.ng,
                      vars.nvar, vars.idens, vars.ixmom, vars.iymom, vars.iener,
                      solid.xl, solid.xr,
                      gamma, U_xl, U_xr)

    _fy = riemannFunc(2, myg.qx, myg.qy, myg.ng,
                      vars.nvar, vars.idens, vars.ixmom, vars.iymom, vars.iener,
                      solid.yl, solid.yr,
                      gamma, U_yl, U_yr)

    F_x = ai.ArrayIndexer(d=_fx, grid=myg)
    F_y = ai.ArrayIndexer(d=_fy, grid=myg)

    tm_riem.end()

    #=========================================================================
    # apply artificial viscosity
    #=========================================================================
    cvisc = rp.get_param("compressible.cvisc")

    _ax, _ay = interface_f.artificial_viscosity(
        myg.qx, myg.qy, myg.ng, myg.dx, myg.dy,
        cvisc, u, v)

    avisco_x = ai.ArrayIndexer(d=_ax, grid=myg)
    avisco_y = ai.ArrayIndexer(d=_ay, grid=myg)


    b = (2,1)

    # F_x = F_x + avisco_x * (U(i-1,j) - U(i,j))
    F_x.v(buf=b, n=vars.idens)[:,:] += \
        avisco_x.v(buf=b)*(dens.ip(-1, buf=b) - dens.v(buf=b))

    F_x.v(buf=b, n=vars.ixmom)[:,:] += \
        avisco_x.v(buf=b)*(xmom.ip(-1, buf=b) - xmom.v(buf=b))

    F_x.v(buf=b, n=vars.iymom)[:,:] += \
        avisco_x.v(buf=b)*(ymom.ip(-1, buf=b) - ymom.v(buf=b))

    F_x.v(buf=b, n=vars.iener)[:,:] += \
        avisco_x.v(buf=b)*(ener.ip(-1, buf=b) - ener.v(buf=b))

    # F_y = F_y + avisco_y * (U(i,j-1) - U(i,j))
    F_y.v(buf=b, n=vars.idens)[:,:] += \
        avisco_y.v(buf=b)*(dens.jp(-1, buf=b) - dens.v(buf=b))

    F_y.v(buf=b, n=vars.ixmom)[:,:] += \
        avisco_y.v(buf=b)*(xmom.jp(-1, buf=b) - xmom.v(buf=b))

    F_y.v(buf=b, n=vars.iymom)[:,:] += \
        avisco_y.v(buf=b)*(ymom.jp(-1, buf=b) - ymom.v(buf=b))

    F_y.v(buf=b, n=vars.iener)[:,:] += \
        avisco_y.v(buf=b)*(ener.jp(-1, buf=b) - ener.v(buf=b))

    tm_flux.end()

    return F_x, F_y
