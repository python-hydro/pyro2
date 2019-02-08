import numpy as np
from numba import njit


@njit(cache=True)
def states(idir, ng, dx, dt,
           ibx, iby, qv, dqv, Bx, By):
    r"""
    predict the cell-centered state to the edges in one-dimension
    using the reconstructed, limited slopes.

    We follow the convection here that ``V_l[i]`` is the left state at the
    i-1/2 interface and ``V_l[i+1]`` is the left state at the i+1/2
    interface.

    The fluxes are going to be defined on the left edge of the
    computational zones::

            |             |             |             |
            |             |             |             |
           -+------+------+------+------+------+------+--
            |     i-1     |      i      |     i+1     |
                         ^ ^           ^
                     q_l,i q_r,i  q_l,i+1

    q_r,i and q_l,i+1 are computed using the information in zone i,j.

    Parameters
    ----------
    idir : int
        Are we predicting to the edges in the x-direction (1) or y-direction (2)?
    ng : int
        The number of ghost cells
    dx : float
        The cell spacing
    dt : float
        The timestep
    irho, iu, iv, ip, ix : int
        Indices of the density, x-velocity, y-velocity, pressure and species in the
        state vector
    naux : int
        The number of species
    gamma : float
        Adiabatic index
    qv : ndarray
        The primitive state vector

    Returns
    -------
    out : ndarray, ndarray
        State vector predicted to the left and right edges
    """

    qx, qy, nvar = qv.shape

    q_l = np.zeros_like(qv)
    q_r = np.zeros_like(qv)

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    # Going to use the simpler method from Stone & Gardiner 09, section 4.2 here

    # compute left- and right-interface states using the monotonized difference in the primitive variables
    for i in range(ilo - 3, ihi + 3):
        for j in range(jlo - 3, jhi + 3):

            q = qv[i, j]
            dq = dqv[i, j]

            if idir == 1:
                q_l[i + 1, j, :] = q + 0.5 * dq
                q_r[i, j, :] = q - 0.5 * dq
                q_l[i + 1, j, ibx] = Bx[i + 1, j]
                q_r[i, j, ibx] = Bx[i, j]
            else:
                q_l[i, j + 1, :] = q + 0.5 * dq
                q_r[i, j, :] = q - 0.5 * dq
                q_l[i, j + 1, iby] = By[i, j + 1]
                q_r[i, j, iby] = By[i, j]

    return q_l, q_r


@njit(cache=True)
def riemann_adiabatic(idir, ng,
                      idens, ixmom, iymom, iener, ixmag, iymag, irhoX, nspec,
                      lower_solid, upper_solid,
                      gamma, U_l, U_r, Bx, By):
    r"""
    HLLE solver for adiabatic magnetohydrodynamics.
    """

    qx, qy, nvar = U_l.shape

    F = np.zeros_like(U_l)

    smallc = 1.e-10
    # smallrho = 1.e-10
    smallp = 1.e-10

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    for i in range(ilo - 2, ihi + 2):
        for j in range(jlo - 2, jhi + 2):
            # primitive variable states
            rho_l = U_l[i, j, idens]

            # un = normal velocity; ut = transverse velocity
            if (idir == 1):
                un_l = U_l[i, j, ixmom] / rho_l
                ut_l = U_l[i, j, iymom] / rho_l
            else:
                un_l = U_l[i, j, iymom] / rho_l
                ut_l = U_l[i, j, ixmom] / rho_l

            if (idir == 1):
                # if we're looking at flux in x-direction, can use x-face centered
                # Bx, but need to get By from U as it's y-face centered
                Bx_l = Bx[i, j]
                Bx_r = Bx[i, j]

                By_l = U_l[i, j, iymag]
                By_r = U_r[i, j, iymag]
            else:
                # the reverse is true for flux in the y-direction
                By_l = By[i, j]
                By_r = By[i, j]
                Bx_l = U_l[i, j, ixmag]
                Bx_r = U_r[i, j, ixmag]

            B2_l = Bx_l**2 + By_l**2
            B2_r = Bx_r**2 + By_r**2

            rhoe_l = U_l[i, j, iener] - 0.5 * rho_l * (un_l**2 + ut_l**2) - \
                0.5 * B2_l

            p_l = rhoe_l * (gamma - 1.0)
            p_l = max(p_l, smallp)

            rho_r = U_r[i, j, idens]

            if (idir == 1):
                un_r = U_r[i, j, ixmom] / rho_r
                ut_r = U_r[i, j, iymom] / rho_r
            else:
                un_r = U_r[i, j, iymom] / rho_r
                ut_r = U_r[i, j, ixmom] / rho_r

            rhoe_r = U_r[i, j, iener] - 0.5 * rho_r * (un_r**2 + ut_r**2) - \
                0.5 * B2_r

            p_r = rhoe_r * (gamma - 1.0)
            p_r = max(p_r, smallp)

            # and the regular sound speeds
            a_l = max(smallc, np.sqrt(gamma * p_l / rho_l))
            a_r = max(smallc, np.sqrt(gamma * p_r / rho_r))

            bx_l = Bx_l / np.sqrt(4 * np.pi)
            bx_r = Bx_r / np.sqrt(4 * np.pi)
            by_l = By_l / np.sqrt(4 * np.pi)
            by_r = By_r / np.sqrt(4 * np.pi)

            # find the Roe average stuff
            # we have to annoyingly do this for the primitive variables then convert back.

            q_av = np.zeros_like(U_l[i, j, :])
            q_av[idens] = np.sqrt(rho_l * rho_r)
            # these are actually the primitive velocities
            q_av[ixmom] = (U_l[i, j, ixmom] / np.sqrt(rho_l) + U_r[i, j, ixmom] / np.sqrt(rho_r)) / \
                (np.sqrt(rho_l) + np.sqrt(rho_r))
            q_av[iymom] = (U_l[i, j, iymom] / np.sqrt(rho_l) + U_r[i, j, iymom] / np.sqrt(rho_r)) / \
                (np.sqrt(rho_l) + np.sqrt(rho_r))

            # this is the enthalpy

            h_l = (gamma * U_l[i, j, iener] -
                   (gamma - 1.0) * (U_l[i, j, ixmom]**2 + U_l[i, j, iymom]**2) / rho_l +
                   0.5 * (2. - gamma) * B2_l) / rho_l

            h_r = (gamma * U_r[i, j, iener] -
                   (gamma - 1.0) * (U_r[i, j, ixmom]**2 + U_r[i, j, iymom]**2) / rho_r +
                   0.5 * (2. - gamma) * B2_r) / rho_r

            q_av[iener] = (h_l * np.sqrt(rho_l) + h_r * np.sqrt(rho_r)) / \
                (np.sqrt(rho_l) + np.sqrt(rho_r))

            q_av[ixmag:iymag + 1] = (U_l[i, j, ixmag:iymag + 1] * np.sqrt(rho_l) +
                                     U_r[i, j, ixmag:iymag + 1] * np.sqrt(rho_r)) / (np.sqrt(rho_l) + np.sqrt(rho_r))

            q_av[irhoX:] = (U_l[i, j, irhoX:] / np.sqrt(rho_l) +
                            U_r[i, j, irhoX:] / np.sqrt(rho_r)) / (np.sqrt(rho_l) + np.sqrt(rho_r))

            U_av = np.zeros_like(q_av)

            U_av[idens] = q_av[idens]
            U_av[ixmom] = q_av[idens] * q_av[ixmom]
            U_av[iymom] = q_av[idens] * q_av[iymom]
            U_av[iener] = (q_av[iener] * q_av[idens] +
                           (gamma - 1) * q_av[idens] * (q_av[ixmom]**2 + q_av[iymom]**2) +
                           0.5 * (gamma - 2.) * (q_av[ixmag]**2 + q_av[iymag]**2)) / gamma
            U_av[ixmag:iymag + 1] = q_av[ixmag:iymag + 1]
            U_av[irhoX:] = q_av[idens] * q_av[irhoX:]

            if idir == 1:
                X = 0.5 * (by_l - by_r)**2 / (np.sqrt(rho_l) + np.sqrt(rho_r))
            else:
                X = 0.5 * (bx_l - bx_r)**2 / (np.sqrt(rho_l) + np.sqrt(rho_r))

            Y = 0.5 * (rho_l + rho_r) / U_av[idens]

            evals = calc_evals(idir, U_av, gamma, idens, ixmom, iymom, iener,
                               ixmag, iymag, irhoX, X, Y)

            # now need to repeat all that stuff to find fast magnetosonic speed
            # in left and right states
            cA2 = (bx_l**2 + by_l**2) / rho_l

            if idir == 1:
                cAx2 = bx_l**2 / rho_l
            else:
                cAx2 = by_l**2 / rho_l

            cf_l = np.sqrt(
                0.5 * (a_l**2 + cA2 + np.sqrt((a_l**2 + cA2)**2 - 4 * a_l**2 * cAx2)))

            cA2 = (bx_r**2 + by_r**2) / rho_r

            if idir == 1:
                cAx2 = bx_r**2 / rho_r
            else:
                cAx2 = by_r**2 / rho_r

            cf_r = np.sqrt(
                0.5 * (a_r**2 + cA2 + np.sqrt((a_r**2 + cA2)**2 - 4 * a_r**2 * cAx2)))

            bp = max(np.max(evals), un_r + cf_r, 0)
            bm = min(np.min(evals), un_l - cf_l, 0)

            # bp = max(un_l + cf_l, un_r + cf_r, 0)
            # bm = min(un_l - cf_l, un_r - cf_r, 0)
            #
            # print("bp, bm = ", bp, bm, min(un_l - cf_l, np.min(evals), 0))

            f_l = consFlux(idir, gamma, idens, ixmom, iymom, iener,
                           ixmag, iymag, irhoX, nspec, U_l[i, j, :])
            f_r = consFlux(idir, gamma, idens, ixmom, iymom, iener,
                           ixmag, iymag, irhoX, nspec, U_r[i, j, :])

            F[i, j, :] = (bp * f_l - bm * f_r +
                          bp * bm * (U_r[i, j, :] - U_l[i, j, :])) / (bp - bm)

    return F


@njit(cache=True)
def riemann_hllc(idir, ng,
                 idens, ixmom, iymom, iener, ixmag, iymag, irhoX, nspec,
                 lower_solid, upper_solid,
                 gamma, U_l, U_r, Bx, By):
    r"""
    This is the HLLC Riemann solver.  The implementation follows
    directly out of Toro's book.  Note: this does not handle the
    transonic rarefaction.

    Parameters
    ----------
    idir : int
        Are we predicting to the edges in the x-direction (1) or y-direction (2)?
    ng : int
        The number of ghost cells
    nspec : int
        The number of species
    idens, ixmom, iymom, iener, irhoX : int
        The indices of the density, x-momentum, y-momentum, internal energy density
        and species partial densities in the conserved state vector.
    lower_solid, upper_solid : int
        Are we at lower or upper solid boundaries?
    gamma : float
        Adiabatic index
    U_l, U_r : ndarray
        Conserved state on the left and right cell edges.

    Returns
    -------
    out : ndarray
        Conserved flux
    """

    qx, qy, nvar = U_l.shape

    F = np.zeros((qx, qy, nvar))

    smallc = 1.e-10
    smallp = 1.e-10

    U_state = np.zeros(nvar)

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    for i in range(ilo - 2, ihi + 2):
        for j in range(jlo - 2, jhi + 2):

            # primitive variable states
            rho_l = U_l[i, j, idens]

            # un = normal velocity; ut = transverse velocity
            if (idir == 1):
                un_l = U_l[i, j, ixmom] / rho_l
                ut_l = U_l[i, j, iymom] / rho_l
            else:
                un_l = U_l[i, j, iymom] / rho_l
                ut_l = U_l[i, j, ixmom] / rho_l

            rhoe_l = U_l[i, j, iener] - 0.5 * rho_l * (un_l**2 + ut_l**2)

            p_l = rhoe_l * (gamma - 1.0)
            p_l = max(p_l, smallp)

            rho_r = U_r[i, j, idens]

            if (idir == 1):
                un_r = U_r[i, j, ixmom] / rho_r
                ut_r = U_r[i, j, iymom] / rho_r
            else:
                un_r = U_r[i, j, iymom] / rho_r
                ut_r = U_r[i, j, ixmom] / rho_r

            rhoe_r = U_r[i, j, iener] - 0.5 * rho_r * (un_r**2 + ut_r**2)

            p_r = rhoe_r * (gamma - 1.0)
            p_r = max(p_r, smallp)

            # compute the sound speeds
            a_l = max(smallc, np.sqrt(gamma * p_l / rho_l))
            a_r = max(smallc, np.sqrt(gamma * p_r / rho_r))

            # Estimate the star quantities -- use one of three methods to
            # do this -- the primitive variable Riemann solver, the two
            # shock approximation, or the two rarefaction approximation.
            # Pick the method based on the pressure states at the
            # interface.

            p_max = max(p_l, p_r)
            p_min = min(p_l, p_r)

            Q = p_max / p_min

            rho_avg = 0.5 * (rho_l + rho_r)
            c_avg = 0.5 * (a_l + a_r)

            # primitive variable Riemann solver (Toro, 9.3)
            factor = rho_avg * c_avg
            # factor2 = rho_avg / c_avg

            pstar = 0.5 * (p_l + p_r) + 0.5 * (un_l - un_r) * factor
            ustar = 0.5 * (un_l + un_r) + 0.5 * (p_l - p_r) / factor

            if (Q > 2 and (pstar < p_min or pstar > p_max)):

                # use a more accurate Riemann solver for the estimate here

                if (pstar < p_min):

                    # 2-rarefaction Riemann solver
                    z = (gamma - 1.0) / (2.0 * gamma)
                    p_lr = (p_l / p_r)**z

                    ustar = (p_lr * un_l / a_l + un_r / a_r +
                             2.0 * (p_lr - 1.0) / (gamma - 1.0)) / \
                            (p_lr / a_l + 1.0 / a_r)

                    pstar = 0.5 * (p_l * (1.0 + (gamma - 1.0) * (un_l - ustar) /
                                          (2.0 * a_l))**(1.0 / z) +
                                   p_r * (1.0 + (gamma - 1.0) * (ustar - un_r) /
                                          (2.0 * a_r))**(1.0 / z))

                else:

                    # 2-shock Riemann solver
                    A_r = 2.0 / ((gamma + 1.0) * rho_r)
                    B_r = p_r * (gamma - 1.0) / (gamma + 1.0)

                    A_l = 2.0 / ((gamma + 1.0) * rho_l)
                    B_l = p_l * (gamma - 1.0) / (gamma + 1.0)

                    # guess of the pressure
                    p_guess = max(0.0, pstar)

                    g_l = np.sqrt(A_l / (p_guess + B_l))
                    g_r = np.sqrt(A_r / (p_guess + B_r))

                    pstar = (g_l * p_l + g_r * p_r -
                             (un_r - un_l)) / (g_l + g_r)

                    ustar = 0.5 * (un_l + un_r) + \
                        0.5 * ((pstar - p_r) * g_r - (pstar - p_l) * g_l)

            # estimate the nonlinear wave speeds

            if (pstar <= p_l):
                # rarefaction
                S_l = un_l - a_l
            else:
                # shock
                S_l = un_l - a_l * np.sqrt(1.0 + ((gamma + 1.0) / (2.0 * gamma)) *
                                           (pstar / p_l - 1.0))

            if (pstar <= p_r):
                # rarefaction
                S_r = un_r + a_r
            else:
                # shock
                S_r = un_r + a_r * np.sqrt(1.0 + ((gamma + 1.0) / (2.0 / gamma)) *
                                           (pstar / p_r - 1.0))

            #  We could just take S_c = u_star as the estimate for the
            #  contact speed, but we can actually do this more accurately
            #  by using the Rankine-Hugonoit jump conditions across each
            #  of the waves (see Toro 10.58, Batten et al. SIAM
            #  J. Sci. and Stat. Comp., 18:1553 (1997)
            S_c = (p_r - p_l + rho_l * un_l * (S_l - un_l) - rho_r * un_r * (S_r - un_r)) / \
                (rho_l * (S_l - un_l) - rho_r * (S_r - un_r))

            # figure out which region we are in and compute the state and
            # the interface fluxes using the HLLC Riemann solver
            if (S_r <= 0.0):
                # R region
                U_state[:] = U_r[i, j, :]

                F[i, j, :] = consFlux(idir, gamma, idens, ixmom, iymom, iener,
                                      ixmag, iymag, irhoX, nspec,
                                      U_state)

            elif (S_r > 0.0 and S_c <= 0):
                # R* region
                HLLCfactor = rho_r * (S_r - un_r) / (S_r - S_c)

                U_state[idens] = HLLCfactor

                if (idir == 1):
                    U_state[ixmom] = HLLCfactor * S_c
                    U_state[iymom] = HLLCfactor * ut_r
                else:
                    U_state[ixmom] = HLLCfactor * ut_r
                    U_state[iymom] = HLLCfactor * S_c

                U_state[iener] = HLLCfactor * (U_r[i, j, iener] / rho_r +
                                               (S_c - un_r) * (S_c + p_r / (rho_r * (S_r - un_r))))

                # species
                if (nspec > 0):
                    U_state[irhoX:irhoX + nspec] = HLLCfactor * \
                        U_r[i, j, irhoX:irhoX + nspec] / rho_r

                # find the flux on the right interface
                F[i, j, :] = consFlux(idir, gamma, idens, ixmom, iymom, iener,
                                      ixmag, iymag, irhoX, nspec,
                                      U_r[i, j, :])

                # correct the flux
                F[i, j, :] = F[i, j, :] + S_r * (U_state[:] - U_r[i, j, :])

            elif (S_c > 0.0 and S_l < 0.0):
                # L* region
                HLLCfactor = rho_l * (S_l - un_l) / (S_l - S_c)

                U_state[idens] = HLLCfactor

                if (idir == 1):
                    U_state[ixmom] = HLLCfactor * S_c
                    U_state[iymom] = HLLCfactor * ut_l
                else:
                    U_state[ixmom] = HLLCfactor * ut_l
                    U_state[iymom] = HLLCfactor * S_c

                U_state[iener] = HLLCfactor * (U_l[i, j, iener] / rho_l +
                                               (S_c - un_l) * (S_c + p_l / (rho_l * (S_l - un_l))))

                # species
                if (nspec > 0):
                    U_state[irhoX:irhoX + nspec] = HLLCfactor * \
                        U_l[i, j, irhoX:irhoX + nspec] / rho_l

                # find the flux on the left interface
                F[i, j, :] = consFlux(idir, gamma, idens, ixmom, iymom, iener,
                                      ixmag, iymag, irhoX, nspec,
                                      U_l[i, j, :])

                # correct the flux
                F[i, j, :] = F[i, j, :] + S_l * (U_state[:] - U_l[i, j, :])

            else:
                # L region
                U_state[:] = U_l[i, j, :]

                F[i, j, :] = consFlux(idir, gamma, idens, ixmom, iymom, iener,
                                      ixmag, iymag, irhoX, nspec,
                                      U_state)

    return F


@njit(cache=True)
def calc_evals(idir, U, gamma, idens, ixmom, iymom, iener, ixmag, iymag, irhoX, X, Y):
    r"""
    Calculate the eigenvalues using section B.3 in Stone, Gardiner et. al 08
    """
    dens = U[idens]
    u = U[ixmom] / U[idens]
    v = U[iymom] / U[idens]
    E = U[iener]
    Bx = U[ixmag]
    By = U[iymag]
    bx = Bx / np.sqrt(4 * np.pi)
    by = By / np.sqrt(4 * np.pi)
    B2 = Bx**2 + By**2
    b2 = bx**2 + by**2
    rhoe = E - 0.5 * dens * (u**2 + v**2) - 0.5 * b2
    P = rhoe * (gamma - 1.0)
    H = (E + P + 0.5 * b2) / dens

    gamma_d = gamma - 1.0
    X_d = (gamma - 2.) * X
    Y_d = (gamma - 2.) * Y

    a2 = gamma_d * (H - 0.5 * (u**2 + v**2) - b2 / dens) - X_d

    if idir == 1:
        CAx2 = bx**2 / dens
        b_norm2 = (gamma_d - Y_d) * by**2
    else:
        CAx2 = by**2 / dens
        b_norm2 = (gamma_d - Y_d) * bx**2

    CA2 = CAx2 + b_norm2 / dens

    # if  4 * a2 * CAx2 > (a2 + CA2)**2:
    #     print("help")

    Cf2 = 0.5 * ((a2 + CA2) + np.sqrt((a2 + CA2)**2 - 4 * a2 * CAx2))
    Cs2 = 0.5 * ((a2 + CA2) - np.sqrt((a2 + CA2)**2 - 4 * a2 * CAx2))

    if idir == 1:
        vx = u
    else:
        vx = v

    evals = np.array([vx - np.sqrt(Cf2), vx - np.sqrt(CAx2), vx - np.sqrt(Cs2),
                      vx, vx + np.sqrt(Cs2), vx + np.sqrt(CAx2), vx + np.sqrt(Cf2)])

    return evals


@njit(cache=True)
def consFlux(idir, gamma, idens, ixmom, iymom, iener, ixmag, iymag, irhoX, naux, U_state):
    r"""
    Calculate the conservative flux.

    Parameters
    ----------
    idir : int
        Are we predicting to the edges in the x-direction (1) or y-direction (2)?
    gamma : float
        Adiabatic index
    idens, ixmom, iymom, iener, ixmag, iymag, irhoX : int
        The indices of the density, x-momentum, y-momentum, x-magnetic field,
        y-magnetic field, internal energy density
        and species partial densities in the conserved state vector.
    naux : int
        The number of species
    U_state : ndarray
        Conserved state vector.

    Returns
    -------
    out : ndarray
        Conserved flux
    """

    F = np.zeros_like(U_state)

    u = U_state[ixmom] / U_state[idens]
    v = U_state[iymom] / U_state[idens]
    bx = U_state[ixmag]
    by = U_state[iymag]
    b2 = bx**2 + by**2

    p = (U_state[iener] - 0.5 * U_state[idens] *
         (u**2 + v**2) - 0.5 * b2) * (gamma - 1.0)

    if (idir == 1):
        F[idens] = U_state[idens] * u
        F[ixmom] = U_state[ixmom] * u + p + 0.5 * b2 - bx**2
        F[iymom] = U_state[iymom] * u - bx * by
        F[iener] = (U_state[iener] + p + 0.5 * b2) * u - \
            bx * (bx * u + by * v)
        F[ixmag] = 0
        F[iymag] = by * u - bx * v

        if (naux > 0):
            F[irhoX:irhoX + naux] = U_state[irhoX:irhoX + naux] * u

    else:
        F[idens] = U_state[idens] * v
        F[ixmom] = U_state[ixmom] * v - bx * by
        F[iymom] = U_state[iymom] * v + p + 0.5 * b2 - by**2
        F[iener] = (U_state[iener] + p + 0.5 * b2) * v - \
            by * (bx * u + by * v)
        F[ixmag] = bx * v - by * u
        F[iymag] = 0

        if (naux > 0):
            F[irhoX:irhoX + naux] = U_state[irhoX:irhoX + naux] * v

    return F


@njit(cache=True)
def emf(ng, idens, ixmom, iymom, iener, ixmag, iymag, irhoX, dx, dy, U, Fx, Fy):
    r"""
    Calculate the EMF at cell corners.

    Eref is the cell-centered reference value used in eq. 81. It can be passed
    in or calculated from the cross product of the velocity and the magnetic
    field in U.

    Note: the slightly messy keyword arguments are to keep numba happy - it
    doesn't like it if a keyword argument is a different type (e.g. None) to
    what it infers the variable to be from elsewhere within the function
    (e.g. an array).
    """

    qx, qy, nvar = U.shape

    Er = np.zeros((qx, qy))  # cell-centered reference emf
    Ex = np.zeros((qx, qy))  # x-edges, (i,j) -> i-1/2, j
    Ey = np.zeros((qx, qy))  # y-edges, (i,j) -> i, j-1/2

    Ec = np.zeros((qx, qy))  # corner, (i,j) -> i-1/2, j-1/2

    dEdy_14 = np.zeros((qx, qy))  # (dE_z / dy)_(i, j-1/4)
    dEdx_14 = np.zeros((qx, qy))  # (dE_z / dx)_(i-1/4, j)

    dEdy_34 = np.zeros((qx, qy))  # (dE_z / dy)_(i, j-3/4)
    dEdx_34 = np.zeros((qx, qy))  # (dE_z / dx)_(i-3/4, j)

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    u = U[:, :, ixmom] / U[:, :, idens]
    v = U[:, :, iymom] / U[:, :, idens]
    bx = U[:, :, ixmag]
    by = U[:, :, iymag]
    Er[:, :] = -(u * by - v * bx)

    # GS05 section 4.1.1
    Ex[:, :] = -Fx[:, :, iymag]
    Ey[:, :] = Fy[:, :, ixmag]

    for i in range(ilo - 3, ihi + 3):
        for j in range(jlo - 3, jhi + 3):

            # get the -1/4 states
            dEdy_14[i, j] = 2 * (Er[i, j] - Ey[i, j]) / dy

            dEdx_14[i, j] = 2 * (Er[i, j] - Ex[i, j]) / dx

            # get the -3/4 states
            dEdy_34[i, j] = 2 * (Ey[i, j] - Er[i, j - 1]) / dy

            dEdx_34[i, j] = 2 * (Ex[i, j] - Er[i - 1, j]) / dx

    for i in range(ilo - 2, ihi + 2):
        for j in range(jlo - 2, jhi + 2):

            # now get the corner states
            # this depends on the sign of the mass flux
            ru = Fx[i, j, idens]  # as Fx(i,j,idens) = (rho * vx)_i-1/2,j
            if ru > 0:
                dEdyx_14 = dEdy_14[i - 1, j]  # dEz/dy_(i-1/2,j-1/4)
            elif ru < 0:
                dEdyx_14 = dEdy_14[i, j]
            else:
                dEdyx_14 = 0.5 * (dEdy_14[i - 1, j] + dEdy_14[i, j])

            # as Fx(i,j-1,idens) = (rho * vx)_i-1/2,j-1
            ru = Fx[i, j - 1, idens]
            if ru > 0:
                dEdyx_34 = dEdy_34[i - 1, j]  # dEz/dy_(i-1/2,j-3/4)
            elif ru < 0:
                dEdyx_34 = dEdy_34[i, j]
            else:
                dEdyx_34 = 0.5 * (dEdy_34[i - 1, j] + dEdy_34[i, j])

            rv = Fy[i, j, idens]  # as Fy(i,j,idens) = (rho * vy)_i,j-1/2
            if rv > 0:
                dEdxy_14 = dEdx_14[i, j - 1]  # dEz/dx_(i-1/4,j-1/2)
            elif rv < 0:
                dEdxy_14 = dEdx_14[i, j]
            else:
                dEdxy_14 = 0.5 * (dEdx_14[i, j - 1] + dEdx_14[i, j])

            # as Fy(i-1,j,idens) = (rho * vy)_i-1,j-1/2
            rv = Fy[i - 1, j, idens]
            if rv > 0:
                dEdxy_34 = dEdx_34[i, j - 1]  # dEz/dx_(i-3/4,j-1/2)
            elif rv < 0:
                dEdxy_34 = dEdx_34[i, j]
            else:
                dEdxy_34 = 0.5 * (dEdx_34[i, j - 1] + dEdx_34[i, j])

            Ec[i, j] = 0.25 * (Ex[i, j] + Ex[i, j - 1] + Ey[i, j] + Ey[i - 1, j]) + \
                0.125 * dy * (dEdyx_14 - dEdyx_34) + \
                0.125 * dx * (dEdxy_14 - dEdxy_34)

    return Ec


# @njit(cache=True)
def sources(idir, ng, idens, ixmom, iymom, iener, ixmag, iymag, irhoX, dx, U, Ux):
    r"""
    Calculate source terms on the idir-interface. U is the cell-centered state,
    Ux should be a state on the idir-interface, where i,j -> i-1/2 ,j (for idir==1).

    Assume Bz = vz = 0 so that iener and iBz sources are 0.
    """
    qx, qy, nvar = U.shape

    S = np.zeros((qx, qy, nvar))

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    for i in range(ilo - ng + 1, ihi + ng - 1):
        for j in range(jlo - ng + 1, jhi + ng - 1):
            if idir == 1:
                S[i, j, ixmom] = U[i, j, ixmag] * \
                    (Ux[i + 1, j, ixmag] - Ux[i, j, ixmag]) / dx
                S[i, j, iymom] = U[i, j, iymag] * \
                    (Ux[i + 1, j, ixmag] - Ux[i, j, ixmag]) / dx
            else:
                S[i, j, ixmom] = U[i, j, ixmag] * \
                    (Ux[i, j + 1, iymag] - Ux[i, j, iymag]) / dx
                S[i, j, iymom] = U[i, j, iymag] * \
                    (Ux[i, j + 1, iymag] - Ux[i, j, iymag]) / dx

    return S

# @njit(cache=True)
# def artificial_viscosity(ng, dx, dy,
#                          cvisc, u, v):
#     r"""
#     Compute the artifical viscosity.  Here, we compute edge-centered
#     approximations to the divergence of the velocity.  This follows
#     directly Colella \ Woodward (1984) Eq. 4.5
#
#     data locations::
#
#         j+3/2--+---------+---------+---------+
#                |         |         |         |
#           j+1  +         |         |         |
#                |         |         |         |
#         j+1/2--+---------+---------+---------+
#                |         |         |         |
#              j +         X         |         |
#                |         |         |         |
#         j-1/2--+---------+----Y----+---------+
#                |         |         |         |
#            j-1 +         |         |         |
#                |         |         |         |
#         j-3/2--+---------+---------+---------+
#                |    |    |    |    |    |    |
#                    i-1        i        i+1
#              i-3/2     i-1/2     i+1/2     i+3/2
#
#     ``X`` is the location of ``avisco_x[i,j]``
#     ``Y`` is the location of ``avisco_y[i,j]``
#
#     Parameters
#     ----------
#     ng : int
#         The number of ghost cells
#     dx, dy : float
#         Cell spacings
#     cvisc : float
#         viscosity parameter
#     u, v : ndarray
#         x- and y-velocities
#
#     Returns
#     -------
#     out : ndarray, ndarray
#         Artificial viscosity in the x- and y-directions
#     """
#
#     qx, qy = u.shape
#
#     avisco_x = np.zeros((qx, qy))
#     avisco_y = np.zeros((qx, qy))
#
#     nx = qx - 2 * ng
#     ny = qy - 2 * ng
#     ilo = ng
#     ihi = ng + nx
#     jlo = ng
#     jhi = ng + ny
#
#     for i in range(ilo - 1, ihi + 1):
#         for j in range(jlo - 1, jhi + 1):
#
#                 # start by computing the divergence on the x-interface.  The
#                 # x-difference is simply the difference of the cell-centered
#                 # x-velocities on either side of the x-interface.  For the
#                 # y-difference, first average the four cells to the node on
#                 # each end of the edge, and: difference these to find the
#                 # edge centered y difference.
#             divU_x = (u[i, j] - u[i - 1, j]) / dx + \
#                 0.25 * (v[i, j + 1] + v[i - 1, j + 1] -
#                         v[i, j - 1] - v[i - 1, j - 1]) / dy
#
#             avisco_x[i, j] = cvisc * max(-divU_x * dx, 0.0)
#
#             # now the y-interface value
#             divU_y = 0.25 * (u[i + 1, j] + u[i + 1, j - 1] - u[i - 1, j] - u[i - 1, j - 1]) / dx + \
#                 (v[i, j] - v[i, j - 1]) / dy
#
#             avisco_y[i, j] = cvisc * max(-divU_y * dy, 0.0)
#
#     return avisco_x, avisco_y
