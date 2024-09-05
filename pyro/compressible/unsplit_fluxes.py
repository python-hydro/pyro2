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

import pyro.compressible as comp
import pyro.compressible.interface as ifc
import pyro.mesh.array_indexer as ai
from pyro.compressible import riemann
from pyro.mesh import reconstruction


def interface_states(my_data, rp, ivars, tc, dt):
    """
    interface_states returns the normal conserved states in the x and y
    interfaces. We get the normal fluxes by finding the normal primitive states,
    Then construct the corresponding conserved states.

    Parameters
    ----------
    my_data : CellCenterData2d object
        The data object containing the grid and advective scalar that
        we are advecting.
    rp : RuntimeParameters object
        The runtime parameters for the simulation
    ivars : Variables object
        The Variables object that tells us which indices refer to which
        variables
    tc : TimerCollection object
        The timers we are using to profile
    dt : float
        The timestep we are advancing through.

    Returns
    -------
    out : ndarray, ndarray, ndarray, ndarray
        Left and right normal conserved states in x and y interfaces
    """

    myg = my_data.grid
    gamma = rp.get_param("eos.gamma")

    # =========================================================================
    # compute the primitive variables
    # =========================================================================
    # Q = (rho, u, v, p, {X})

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

    # monotonized central differences
    tm_limit = tc.timer("limiting")
    tm_limit.begin()

    limiter = rp.get_param("compressible.limiter")

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

    V_l, V_r = ifc.states(1, myg.ng, myg.Lx, dt,
                          ivars.irho, ivars.iu, ivars.iv, ivars.ip, ivars.ix,
                          ivars.naux,
                          gamma,
                          q, ldx)

    tm_states.end()

    # transform primitive interface states back into conserved variables
    U_xl = comp.prim_to_cons(V_l, gamma, ivars, myg)
    U_xr = comp.prim_to_cons(V_r, gamma, ivars, myg)

    # =========================================================================
    # y-direction
    # =========================================================================

    # left and right primitive variable states
    tm_states.begin()

    _V_l, _V_r = ifc.states(2, myg.ng, myg.Ly, dt,
                            ivars.irho, ivars.iu, ivars.iv, ivars.ip, ivars.ix,
                            ivars.naux,
                            gamma,
                            q, ldy)
    V_l = ai.ArrayIndexer(d=_V_l, grid=myg)
    V_r = ai.ArrayIndexer(d=_V_r, grid=myg)

    tm_states.end()

    # transform primitive interface states back into conserved variables
    U_yl = comp.prim_to_cons(V_l, gamma, ivars, myg)
    U_yr = comp.prim_to_cons(V_r, gamma, ivars, myg)

    return U_xl, U_xr, U_yl, U_yr


def apply_source_terms(U_xl, U_xr, U_yl, U_yr,
                       my_data, my_aux, rp, ivars, tc, dt):
    """
    This function applies source terms including external (gravity),
    geometric terms, and pressure terms to the left and right
    interface states (normal conserved states).
    Both geometric and pressure terms arise purely from geometry.

    Parameters
    ----------
    U_xl, U_xr, U_yl, U_yr: ndarray, ndarray, ndarray, ndarray
        Conserved states in the left and right x-interface
        and left and right y-interface.
    my_data : CellCenterData2d object
        The data object containing the grid and advective scalar that
        we are advecting.
    my_aux : CellCenterData2d object
        The data object that carries auxiliary quantities which we need
        to fill in the ghost cells.
    rp : RuntimeParameters object
        The runtime parameters for the simulation
    ivars : Variables object
        The Variables object that tells us which indices refer to which
        variables
    tc : TimerCollection object
        The timers we are using to profile
    dt : float
        The timestep we are advancing through.

    Returns
    -------
    out : ndarray, ndarray, ndarray, ndarray
        Left and right normal conserved states in x and y interfaces
        with source terms added.
    """

    tm_source = tc.timer("sourceTerms")
    tm_source.begin()

    myg = my_data.grid

    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    pres = my_data.get_var("pressure")

    dens_src = my_aux.get_var("dens_src")
    xmom_src = my_aux.get_var("xmom_src")
    ymom_src = my_aux.get_var("ymom_src")
    E_src = my_aux.get_var("E_src")

    grav = rp.get_param("compressible.grav")

    # Calculate external source (gravity), geometric, and pressure terms
    if myg.coord_type == 1:
        # assume gravity points in r-direction in spherical.
        dens_src.v()[:, :] = 0.0
        xmom_src.v()[:, :] = dens.v()*grav + \
            ymom.v()**2 / (dens.v()*myg.x2d.v()) - \
            (pres.ip(1) - pres.v()) / myg.Lx.v()
        ymom_src.v()[:, :] = -(pres.jp(1) - pres.v()) / myg.Ly.v() - \
            xmom.v()*ymom.v() / (dens.v()*myg.x2d.v())
        E_src.v()[:, :] = xmom.v()*grav

    else:
        # assume gravity points in y-direction in cartesian
        dens_src.v()[:, :] = 0.0
        xmom_src.v()[:, :] = 0.0
        ymom_src.v()[:, :] = dens.v()*grav
        E_src.v()[:, :] = ymom.v()*grav

    my_aux.fill_BC("dens_src")
    my_aux.fill_BC("xmom_src")
    my_aux.fill_BC("ymom_src")
    my_aux.fill_BC("E_src")

    # U_xl[i,j] += 0.5*dt*source[i-1, j]
    U_xl.v(buf=1, n=ivars.ixmom)[:, :] += 0.5*dt*xmom_src.ip(-1, buf=1)
    U_xl.v(buf=1, n=ivars.iymom)[:, :] += 0.5*dt*ymom_src.ip(-1, buf=1)
    U_xl.v(buf=1, n=ivars.iener)[:, :] += 0.5*dt*E_src.ip(-1, buf=1)

    # U_xr[i,j] += 0.5*dt*source[i, j]
    U_xr.v(buf=1, n=ivars.ixmom)[:, :] += 0.5*dt*xmom_src.v(buf=1)
    U_xr.v(buf=1, n=ivars.iymom)[:, :] += 0.5*dt*ymom_src.v(buf=1)
    U_xr.v(buf=1, n=ivars.iener)[:, :] += 0.5*dt*E_src.v(buf=1)

    # U_yl[i,j] += 0.5*dt*source[i, j-1]
    U_yl.v(buf=1, n=ivars.ixmom)[:, :] += 0.5*dt*xmom_src.jp(-1, buf=1)
    U_yl.v(buf=1, n=ivars.iymom)[:, :] += 0.5*dt*ymom_src.jp(-1, buf=1)
    U_yl.v(buf=1, n=ivars.iener)[:, :] += 0.5*dt*E_src.jp(-1, buf=1)

    # U_yr[i,j] += 0.5*dt*source[i, j]
    U_yr.v(buf=1, n=ivars.ixmom)[:, :] += 0.5*dt*xmom_src.v(buf=1)
    U_yr.v(buf=1, n=ivars.iymom)[:, :] += 0.5*dt*ymom_src.v(buf=1)
    U_yr.v(buf=1, n=ivars.iener)[:, :] += 0.5*dt*E_src.v(buf=1)

    tm_source.end()

    return U_xl, U_xr, U_yl, U_yr


def apply_transverse_flux(U_xl, U_xr, U_yl, U_yr,
                          my_data, rp, ivars, solid, tc, dt):
    """
    This function applies transverse correction terms to the
    normal conserved states after applying other source terms.

    We construct the state perpendicular to the interface
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

    Parameters
    ----------
    U_xl, U_xr, U_yl, U_yr: ndarray, ndarray, ndarray, ndarray
        Conserved states in the left and right x-interface
        and left and right y-interface.
    my_data : CellCenterData2d object
        The data object containing the grid and advective scalar that
        we are advecting.
    rp : RuntimeParameters object
        The runtime parameters for the simulation
    ivars : Variables object
        The Variables object that tells us which indices refer to which
        variables
    solid: A container class
        This is used in Riemann solver to indicate which side has solid boundary
    tc : TimerCollection object
        The timers we are using to profile
    dt : float
        The timestep we are advancing through.

    Returns
    -------
    out : ndarray, ndarray, ndarray, ndarray
        Left and right normal conserved states in x and y interfaces
        with source terms added.
    """

    # Use Riemann Solver to get interface flux using the left and right states

    F_x = riemann.riemann_flux(1, U_xl, U_xr,
                               my_data, rp, ivars,
                               solid.xl, solid.xr, tc)

    F_y = riemann.riemann_flux(2, U_yl, U_yr,
                               my_data, rp, ivars,
                               solid.yl, solid.yr, tc)

    # Now we update the conserved states using the transverse fluxes.

    myg = my_data.grid

    tm_transverse = tc.timer("transverse flux addition")
    tm_transverse.begin()

    b = (2, 1)
    hdtV = 0.5*dt / myg.V

    for n in range(ivars.nvar):

        # U_xl[i,j,:] = U_xl[i,j,:] - 0.5*dt/dy * (F_y[i-1,j+1,:] - F_y[i-1,j,:])
        U_xl.v(buf=b, n=n)[:, :] += \
            - hdtV.v(buf=b)*(F_y.ip_jp(-1, 1, buf=b, n=n)*myg.Ay.ip_jp(-1, 1, buf=b) -
                             F_y.ip(-1, buf=b, n=n)*myg.Ay.ip(-1, buf=b))

        # U_xr[i,j,:] = U_xr[i,j,:] - 0.5*dt/dy * (F_y[i,j+1,:] - F_y[i,j,:]
        U_xr.v(buf=b, n=n)[:, :] += \
            - hdtV.v(buf=b)*(F_y.jp(1, buf=b, n=n)*myg.Ay.jp(1, buf=b) -
                             F_y.v(buf=b, n=n)*myg.Ay.v(buf=b))

        # U_yl[i,j,:] = U_yl[i,j,:] - 0.5*dt/dx * (F_x[i+1,j-1,:] - F_x[i,j-1,:])
        U_yl.v(buf=b, n=n)[:, :] += \
            - hdtV.v(buf=b)*(F_x.ip_jp(1, -1, buf=b, n=n)*myg.Ax.ip_jp(1, -1, buf=b) -
                             F_x.jp(-1, buf=b, n=n)*myg.Ax.jp(-1, buf=b))

        # U_yr[i,j,:] = U_yr[i,j,:] - 0.5*dt/dx * (F_x[i+1,j,:] - F_x[i,j,:])
        U_yr.v(buf=b, n=n)[:, :] += \
            - hdtV.v(buf=b)*(F_x.ip(1, buf=b, n=n)*myg.Ax.ip(1, buf=b) -
                             F_x.v(buf=b, n=n)*myg.Ax.v(buf=b))

    tm_transverse.end()

    return U_xl, U_xr, U_yl, U_yr


def apply_artificial_viscosity(F_x, F_y, q,
                               my_data, rp, ivars):
    """
    This applies artificial viscosity correction terms to the fluxes.

    Parameters
    ----------
    F_x, F_y : ndarray, ndarray
        Fluxes in x and y interface.
    q : ndarray
        Primitive variables
    my_data : CellCenterData2d object
        The data object containing the grid and advective scalar that
        we are advecting.
    rp : RuntimeParameters object
        The runtime parameters for the simulation
    ivars : Variables object
        The Variables object that tells us which indices refer to which
        variables
    dt : float
        The timestep we are advancing through.

    Returns
    -------
    out : ndarray, ndarray
        Fluxes in x and y interface.
    """

    cvisc = rp.get_param("compressible.cvisc")

    myg = my_data.grid

    _ax, _ay = ifc.artificial_viscosity(myg.ng, myg.dx, myg.dy, myg.Lx, myg.Ly,
                                        myg.xmin, myg.ymin, myg.coord_type,
        cvisc, q.v(n=ivars.iu, buf=myg.ng), q.v(n=ivars.iv, buf=myg.ng))

    avisco_x = ai.ArrayIndexer(d=_ax, grid=myg)
    avisco_y = ai.ArrayIndexer(d=_ay, grid=myg)

    b = (2, 1)

    for n in range(ivars.nvar):
        # F_x = F_x + avisco_x * (U(i-1,j) - U(i,j))
        var = my_data.get_var_by_index(n)

        F_x.v(buf=b, n=n)[:, :] += \
            avisco_x.v(buf=b)*(var.ip(-1, buf=b) - var.v(buf=b))

        # F_y = F_y + avisco_y * (U(i,j-1) - U(i,j))
        F_y.v(buf=b, n=n)[:, :] += \
            avisco_y.v(buf=b)*(var.jp(-1, buf=b) - var.v(buf=b))

    return F_x, F_y
