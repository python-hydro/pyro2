"""Implementation of the Colella 2nd order unsplit Godunov scheme.  This
is a 2-dimensional implementation only.  We assume that the grid is
uniform, but it is relatively straightforward to relax this
assumption.

There are several different options for this solver (they are all
discussed in the Colella paper).

* limiter: 0 = no limiting; 1 = 2nd order MC limiter; 2 = 4th order MC limiter

* riemann: HLLC or Roe-fix

* use_flattening: set to 1 to use the multidimensional flattening at shocks

* delta, z0, z1: flattening parameters (we use Colella 1990 defaults)

The grid indices look like::

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

.. math::

   U_t + F^x_x + F^y_y = H

we want U_{i+1/2}^{n+1/2} -- the interface values that are input to
the Riemann problem through the faces for each zone.

Taylor expanding yields::

    n+1/2                     dU           dU
   U          = U   + 0.5 dx  --  + 0.5 dt --
    i+1/2,j,L    i,j          dx           dt


                              dU             dF^x   dF^y
              = U   + 0.5 dx  --  - 0.5 dt ( ---- + ---- - H )
                 i,j          dx              dx     dy


                               dU      dF^x            dF^y
              = U   + 0.5 ( dx -- - dt ---- ) - 0.5 dt ---- + 0.5 dt H
                 i,j           dx       dx              dy


                                   dt       dU           dF^y
              = U   + 0.5 dx ( 1 - -- A^x ) --  - 0.5 dt ---- + 0.5 dt H
                 i,j               dx       dx            dy


                                 dt       _            dF^y
              = U   + 0.5  ( 1 - -- A^x ) DU  - 0.5 dt ---- + 0.5 dt H
                 i,j             dx                     dy

                      +----------+-----------+  +----+----+   +---+---+
                                 |                   |            |

                     this is the monotonized   this is the   source term
                     central difference term   transverse
                                               flux term

There are two components, the central difference in the normal to the
interface, and the transverse flux difference.  This is done for the
left and right sides of all 4 interfaces in a zone, which are then
used as input to the Riemann problem, yielding the 1/2 time interface
values::

    n+1/2
   U
    i+1/2,j

Then, the zone average values are updated in the usual finite-volume
way::

    n+1    n     dt    x  n+1/2       x  n+1/2
   U    = U    + -- { F (U       ) - F (U       ) }
    i,j    i,j   dx       i-1/2,j        i+1/2,j


                 dt    y  n+1/2       y  n+1/2
               + -- { F (U       ) - F (U       ) }
                 dy       i,j-1/2        i,j+1/2

Updating U_{i,j}:

* We want to find the state to the left and right (or top and bottom)
  of each interface, ex. U_{i+1/2,j,[lr]}^{n+1/2}, and use them to
  solve a Riemann problem across each of the four interfaces.

* U_{i+1/2,j,[lr]}^{n+1/2} is comprised of two parts, the computation
  of the monotonized central differences in the normal direction
  (eqs. 2.8, 2.10) and the computation of the transverse derivatives,
  which requires the solution of a Riemann problem in the transverse
  direction (eqs. 2.9, 2.14).

  * the monotonized central difference part is computed using the
    primitive variables.

  * We compute the central difference part in both directions before
    doing the transverse flux differencing, since for the high-order
    transverse flux implementation, we use these as the input to the
    transverse Riemann problem.

"""

import pyro.mesh.array_indexer as ai
import pyro.swe as comp
import pyro.swe.interface as ifc
from pyro.mesh import reconstruction
from pyro.util import msg


def unsplit_fluxes(my_data, rp, ivars, solid, tc, dt):
    """
    unsplitFluxes returns the fluxes through the x and y interfaces by
    doing an unsplit reconstruction of the interface values and then
    solving the Riemann problem through all the interfaces at once

    The runtime parameter g is assumed to be the gravitational
    acceleration in the y-direction

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
    dt : float
        The timestep we are advancing through.

    Returns
    -------
    out : ndarray, ndarray
        The fluxes on the x- and y-interfaces

    """

    tm_flux = tc.timer("unsplitFluxes")
    tm_flux.begin()

    myg = my_data.grid

    g = rp.get_param("swe.grav")

    # =========================================================================
    # compute the primitive variables
    # =========================================================================
    # Q = (h, u, v, {X})

    q = comp.cons_to_prim(my_data.data, ivars, myg)

    # =========================================================================
    # compute the flattening coefficients
    # =========================================================================

    # there is a single flattening coefficient (xi) for all directions
    use_flattening = rp.get_param("swe.use_flattening")

    if use_flattening:
        xi_x = reconstruction.flatten(myg, q, 1, ivars, rp)
        xi_y = reconstruction.flatten(myg, q, 2, ivars, rp)

        xi = reconstruction.flatten_multid(myg, q, xi_x, xi_y, ivars)
    else:
        xi = 1.0

    # monotonized central differences
    tm_limit = tc.timer("limiting")
    tm_limit.begin()

    limiter = rp.get_param("swe.limiter")

    ldx = myg.scratch_array(nvar=ivars.nvar)
    ldy = myg.scratch_array(nvar=ivars.nvar)

    for n in range(ivars.nvar):
        ldx[:, :, n] = xi*reconstruction.limit(q[:, :, n], myg, 1, limiter)
        ldy[:, :, n] = xi*reconstruction.limit(q[:, :, n], myg, 2, limiter)

    tm_limit.end()

    # =========================================================================
    # x-direction
    # =========================================================================

    # left and right primitive variable states
    tm_states = tc.timer("interfaceStates")
    tm_states.begin()

    V_l, V_r = ifc.states(1, myg.ng, myg.dx, dt,
                          ivars.ih, ivars.iu, ivars.iv, ivars.ix,
                          ivars.naux,
                          g,
                          q, ldx)

    tm_states.end()

    # transform interface states back into conserved variables
    U_xl = comp.prim_to_cons(V_l, ivars, myg)
    U_xr = comp.prim_to_cons(V_r, ivars, myg)

    # =========================================================================
    # y-direction
    # =========================================================================

    # left and right primitive variable states
    tm_states.begin()

    _V_l, _V_r = ifc.states(2, myg.ng, myg.dy, dt,
                            ivars.ih, ivars.iu, ivars.iv, ivars.ix,
                            ivars.naux,
                            g,
                            q, ldy)
    V_l = ai.ArrayIndexer(d=_V_l, grid=myg)
    V_r = ai.ArrayIndexer(d=_V_r, grid=myg)

    tm_states.end()

    # transform interface states back into conserved variables
    U_yl = comp.prim_to_cons(V_l, ivars, myg)
    U_yr = comp.prim_to_cons(V_r, ivars, myg)

    # =========================================================================
    # compute transverse fluxes
    # =========================================================================
    tm_riem = tc.timer("riemann")
    tm_riem.begin()

    riemann = rp.get_param("swe.riemann")

    riemannFunc = None
    if riemann == "HLLC":
        riemannFunc = ifc.riemann_hllc
    elif riemann == "Roe":
        riemannFunc = ifc.riemann_roe
    else:
        msg.fail("ERROR: Riemann solver undefined")

    _fx = riemannFunc(1, myg.ng,
                      ivars.ih, ivars.ixmom, ivars.iymom, ivars.ihx, ivars.naux,
                      solid.xl, solid.xr,
                      g, U_xl, U_xr)

    _fy = riemannFunc(2, myg.ng,
                      ivars.ih, ivars.ixmom, ivars.iymom, ivars.ihx, ivars.naux,
                      solid.yl, solid.yr,
                      g, U_yl, U_yr)

    F_x = ai.ArrayIndexer(d=_fx, grid=myg)
    F_y = ai.ArrayIndexer(d=_fy, grid=myg)

    tm_riem.end()

    # =========================================================================
    # construct the interface values of U now
    # =========================================================================

    """
    finally, we can construct the state perpendicular to the interface
    by adding the central difference part to the transverse flux
    difference.

    The states that we represent by indices i,j are shown below
    (1,2,3,4):


      j+3/2--+----------+----------+----------+
             |          |          |          |
             |          |          |          |
        j+1 -+          |          |          |
             |          |          |          |
             |          |          |          |    1: U_xl[i,j,:] = U
      j+1/2--+----------XXXXXXXXXXXX----------+                      i-1/2,j,L
             |          X          X          |
             |          X          X          |
          j -+        1 X 2        X          |    2: U_xr[i,j,:] = U
             |          X          X          |                      i-1/2,j,R
             |          X    4     X          |
      j-1/2--+----------XXXXXXXXXXXX----------+
             |          |    3     |          |    3: U_yl[i,j,:] = U
             |          |          |          |                      i,j-1/2,L
        j-1 -+          |          |          |
             |          |          |          |
             |          |          |          |    4: U_yr[i,j,:] = U
      j-3/2--+----------+----------+----------+                      i,j-1/2,R
             |    |     |    |     |    |     |
                 i-1         i         i+1
           i-3/2      i-1/2      i+1/2      i+3/2


    remember that the fluxes are stored on the left edge, so

    F_x[i,j,:] = F_x
                    i-1/2, j

    F_y[i,j,:] = F_y
                    i, j-1/2

    """

    tm_transverse = tc.timer("transverse flux addition")
    tm_transverse.begin()

    dtdx = dt/myg.dx
    dtdy = dt/myg.dy

    b = (2, 1)

    for n in range(ivars.nvar):

        # U_xl[i,j,:] = U_xl[i,j,:] - 0.5*dt/dy * (F_y[i-1,j+1,:] - F_y[i-1,j,:])
        U_xl.v(buf=b, n=n)[:, :] += \
            - 0.5*dtdy*(F_y.ip_jp(-1, 1, buf=b, n=n) - F_y.ip(-1, buf=b, n=n))

        # U_xr[i,j,:] = U_xr[i,j,:] - 0.5*dt/dy * (F_y[i,j+1,:] - F_y[i,j,:])
        U_xr.v(buf=b, n=n)[:, :] += \
            - 0.5*dtdy*(F_y.jp(1, buf=b, n=n) - F_y.v(buf=b, n=n))

        # U_yl[i,j,:] = U_yl[i,j,:] - 0.5*dt/dx * (F_x[i+1,j-1,:] - F_x[i,j-1,:])
        U_yl.v(buf=b, n=n)[:, :] += \
            - 0.5*dtdx*(F_x.ip_jp(1, -1, buf=b, n=n) - F_x.jp(-1, buf=b, n=n))

        # U_yr[i,j,:] = U_yr[i,j,:] - 0.5*dt/dx * (F_x[i+1,j,:] - F_x[i,j,:])
        U_yr.v(buf=b, n=n)[:, :] += \
            - 0.5*dtdx*(F_x.ip(1, buf=b, n=n) - F_x.v(buf=b, n=n))

    tm_transverse.end()

    # =========================================================================
    # construct the fluxes normal to the interfaces
    # =========================================================================

    # up until now, F_x and F_y stored the transverse fluxes, now we
    # overwrite with the fluxes normal to the interfaces

    tm_riem.begin()

    _fx = riemannFunc(1, myg.ng,
                      ivars.ih, ivars.ixmom, ivars.iymom, ivars.ihx, ivars.naux,
                      solid.xl, solid.xr,
                      g, U_xl, U_xr)

    _fy = riemannFunc(2, myg.ng,
                      ivars.ih, ivars.ixmom, ivars.iymom, ivars.ihx, ivars.naux,
                      solid.yl, solid.yr,
                      g, U_yl, U_yr)

    F_x = ai.ArrayIndexer(d=_fx, grid=myg)
    F_y = ai.ArrayIndexer(d=_fy, grid=myg)

    tm_riem.end()

    tm_flux.end()

    return F_x, F_y
