"""Implementation of the Colella 2nd order unsplit Godunov scheme.  This
is a 2-dimensional implementation only.  We assume that the grid is
uniform, but it is relatively straightforward to relax this
assumption.

There are several different options for this solver (they are all
discussed in the Colella paper).

* limiter: 0 = no limiting; 1 = 2nd order MC limiter; 2 = 4th order MC limiter

* riemann: HLLC or CGF (for Colella, Glaz, and Freguson solver)

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

import mhd.interface as ifc
import mhd as comp
import mesh.array_indexer as ai
import numpy as np


def timestep(cc_data, fcx_data, fcy_data, my_aux, rp, ivars, solid, tc, dt):
    """
    timestep evolves the *_data through a single timestep dt

    currently we assume a gamma-law EOS

    The runtime parameter grav is assumed to be the gravitational
    acceleration in the y-direction

    Parameters
    ----------
    cc_data : CellCenterData2d object
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

    myg = cc_data.grid

    gamma = rp.get_param("eos.gamma")

    buf = 2

    # =========================================================================
    # compute the primitive variables
    # =========================================================================
    # Q = (rho, u, v, p, {X})
    U = cc_data.data

    U_old = myg.scratch_array(nvar=ivars.nvar)
    U_old[:, :, :] = U

    # get face-centered magnetic field components
    Bx = fcx_data.get_var("x-magnetic-field")
    By = fcy_data.get_var("y-magnetic-field")

    Bx_old = np.zeros_like(Bx.v(buf=buf))
    By_old = np.zeros_like(By.v(buf=buf))

    Bx_old[:, :] = Bx.v(buf=buf)
    By_old[:, :] = By.v(buf=buf)

    ############################################################################
    # STEP 1. Using a Riemann solver, construct first order upwind fluxes.
    ############################################################################

    riemannFunc = ifc.riemann_adiabatic

    U_xl = myg.scratch_array(nvar=ivars.nvar)
    U_xr = myg.scratch_array(nvar=ivars.nvar)

    U_yl = myg.scratch_array(nvar=ivars.nvar)
    U_yr = myg.scratch_array(nvar=ivars.nvar)

    # =========================================================================
    # x-direction
    # =========================================================================
    U_xl[1:-1, :, :] = U[:-2, :, :]
    U_xl[:, :, ivars.ixmag] = Bx[:-1, :]
    U_xr[:, :, :] = U
    U_xr[:, :, ivars.ixmag] = Bx[:-1, :]

    _fx = riemannFunc(1, myg.ng,
                      ivars.idens, ivars.ixmom, ivars.iymom, ivars.iener,
                      ivars.ixmag, ivars.iymag, ivars.irhox, ivars.naux,
                      solid.xl, solid.xr,
                      gamma, U_xl, U_xr, Bx, By)

    # =========================================================================
    # y-direction
    # =========================================================================
    U_yl[:, 1:-1, :] = U[:, :-2, :]
    U_yl[:, :, ivars.iymag] = By[:, :-1]
    U_yr[:, :, :] = U
    U_yr[:, :, ivars.iymag] = By[:, :-1]

    _fy = riemannFunc(2, myg.ng,
                      ivars.idens, ivars.ixmom, ivars.iymom, ivars.iener,
                      ivars.ixmag, ivars.iymag, ivars.irhox, ivars.naux,
                      solid.yl, solid.yr,
                      gamma, U_yl, U_yr, Bx, By)

    F_x = ai.ArrayIndexer(d=_fx, grid=myg)
    F_y = ai.ArrayIndexer(d=_fy, grid=myg)

    ############################################################################
    # STEP 2. Calculate the CT electric fields at cell corners
    ############################################################################

    _emf = ifc.emf(myg.ng, ivars.idens, ivars.ixmom, ivars.iymom, ivars.iener,
                   ivars.ixmag, ivars.iymag, ivars.irhox,
                   myg.dx, myg.dy, U, F_x, F_y)

    emf = ai.ArrayIndexer(d=_emf, grid=myg)

    ############################################################################
    # STEP 3. Update the cell-centered hydrodynamical variables for one half
    # time step using flux differences in all direcitons. Update the
    # face-centered components of the magnetic field for one half time step
    # using CT.
    ############################################################################

    dtdx = dt / myg.dx
    dtdy = dt / myg.dy

    for n in range(ivars.nvar):

        if n == ivars.ixmag or n == ivars.iymag:
            continue

        U.v(buf=buf, n=n)[:, :] += -0.5 * dtdx * (F_x.ip(1, buf=buf, n=n) -
                                                  F_x.v(buf=buf, n=n)) - \
            0.5 * dtdy * (F_y.jp(1, buf=buf, n=n) - F_y.v(buf=buf, n=n))

    Bx.v(buf=buf)[:-1, :] -= 0.5 * dtdy * (emf.jp(1, buf=buf) - emf.v(buf=buf))
    By.v(buf=buf)[:, :-1] += 0.5 * dtdx * (emf.ip(1, buf=buf) - emf.v(buf=buf))

    ############################################################################
    # STEP 4. Compute the cell-centered magnetic field at the half time step
    # from the average of the face-centered field computed in step 3.
    ############################################################################

    U.v(buf=buf, n=ivars.ixmag)[:, :] = 0.5 * \
        (Bx.ip(1, buf=buf)[:-1, :] + Bx.v(buf=buf)[:-1, :])
    U.v(buf=buf, n=ivars.iymag)[:, :] = 0.5 * \
        (By.jp(1, buf=buf)[:, :-1] + By.v(buf=buf)[:, :-1])

    ############################################################################
    # STEP 5. Compute the left- and right-state quantities at the half time step
    # at cell interfaces
    ############################################################################

    # calculate primitive variables at half time step from updated U
    q = comp.cons_to_prim(U, gamma, ivars, myg)

    # =========================================================================
    # x-direction
    # =========================================================================

    # left and right primitive variable states
    tm_states = tc.timer("interfaceStates")
    tm_states.begin()

    V_l, V_r = ifc.states(1, myg.ng, myg.dx, dt,
                          ivars.irho, ivars.iu, ivars.iv, ivars.ip,
                          ivars.ibx, ivars.iby, ivars.ix,
                          ivars.naux,
                          gamma,
                          q, Bx, By)

    tm_states.end()

    # transform interface states back into conserved variables
    U_xl = comp.prim_to_cons(V_l, gamma, ivars, myg)
    U_xr = comp.prim_to_cons(V_r, gamma, ivars, myg)

    # =========================================================================
    # y-direction
    # =========================================================================

    # left and right primitive variable states
    tm_states.begin()

    _V_l, _V_r = ifc.states(2, myg.ng, myg.dy, dt,
                            ivars.irho, ivars.iu, ivars.iv, ivars.ip,
                            ivars.ibx, ivars.iby, ivars.ix,
                            ivars.naux,
                            gamma,
                            q, Bx, By)

    V_l = ai.ArrayIndexer(d=_V_l, grid=myg)
    V_r = ai.ArrayIndexer(d=_V_r, grid=myg)

    tm_states.end()

    # transform interface states back into conserved variables
    U_yl = comp.prim_to_cons(V_l, gamma, ivars, myg)
    U_yr = comp.prim_to_cons(V_r, gamma, ivars, myg)

    ############################################################################
    # STEP 6. Using a Riemann solver, construct 1d fluces at interfaces.
    ############################################################################

    _fx = riemannFunc(1, myg.ng,
                      ivars.idens, ivars.ixmom, ivars.iymom, ivars.iener,
                      ivars.ixmag, ivars.iymag, ivars.irhox, ivars.naux,
                      solid.xl, solid.xr,
                      gamma, U_xl, U_xr, Bx, By)

    _fy = riemannFunc(2, myg.ng,
                      ivars.idens, ivars.ixmom, ivars.iymom, ivars.iener,
                      ivars.ixmag, ivars.iymag, ivars.irhox, ivars.naux,
                      solid.yl, solid.yr,
                      gamma, U_yl, U_yr, Bx, By)

    F_x = ai.ArrayIndexer(d=_fx, grid=myg)
    F_y = ai.ArrayIndexer(d=_fy, grid=myg)

    ############################################################################
    # STEP 7. Compute cell-centered reference EMF at half time step using cell-
    # centered velocities and magnetic field computed in steps 3 and 4. Then
    # calculated the CT emf at cell corners from components of face-centered
    # fluxes found in step 6 and the reference field.
    ############################################################################

    # cell-centered velocites at half time
    u_half = U.v(n=ivars.ixmom, buf=buf) / U.v(n=ivars.idens, buf=buf)
    v_half = U.v(n=ivars.iymom, buf=buf) / U.v(n=ivars.idens, buf=buf)

    E_cc_half = myg.scratch_array()
    E_cc_half.v(buf=buf)[:, :] = - \
        (u_half * By.v(buf=buf)[:, :-1] - v_half * Bx.v(buf=buf)[:-1, :])

    _emf = ifc.emf(myg.ng, ivars.idens, ivars.ixmom, ivars.iymom, ivars.iener,
                   ivars.ixmag, ivars.iymag, ivars.irhox,
                   myg.dx, myg.dy, U, F_x, F_y, E_cc_half, True)

    emf = ai.ArrayIndexer(d=_emf, grid=myg)

    ############################################################################
    # STEP 8. Update the cell-centered hydodynamical variables for a full
    # timestep. Update the face-centered components of the magnetic field for a
    # full timestep using CT and the emfs from step 7.
    ############################################################################

    for n in range(ivars.nvar):

        if n == ivars.ixmag or n == ivars.iymag:
            continue

        U.v(buf=buf, n=n)[:, :] = U_old.v(buf=buf, n=n) - \
            dtdx * (F_x.ip(1, buf=buf, n=n) - F_x.v(buf=buf, n=n)) - \
            dtdy * (F_y.jp(1, buf=buf, n=n) - F_y.v(buf=buf, n=n))

    Bx.v(buf=buf)[:-1, :] = Bx_old[:-1, :] - \
        dtdy * (emf.jp(1, buf=buf) - emf.v(buf=buf))
    By.v(buf=buf)[:, :-1] = By_old[:, :-1] + \
        dtdx * (emf.ip(1, buf=buf) - emf.v(buf=buf))

    ############################################################################
    # STEP 9. Compute the cell-centered magnetic field from the updated
    # face-centrered values.
    ############################################################################

    U.v(buf=buf, n=ivars.ixmag)[:, :] = 0.5 * \
        (Bx.ip(1, buf=buf)[:-1, :] + Bx.v(buf=buf)[:-1, :])
    U.v(buf=buf, n=ivars.iymag)[:, :] = 0.5 * \
        (By.jp(1, buf=buf)[:, :-1] + By.v(buf=buf)[:, :-1])
