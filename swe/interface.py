import numpy as np
from numba import njit


@njit(cache=True)
def states(idir, ng, dx, dt,
           ih, iu, iv, ix, nspec,
           g,
           qv, dqv):
    r"""
    predict the cell-centered state to the edges in one-dimension
    using the reconstructed, limited slopes.

    We follow the convection here that ``V_l[i]`` is the left state at the
    i-1/2 interface and ``V_l[i+1]`` is the left state at the i+1/2
    interface.


    We need the left and right eigenvectors and the eigenvalues for
    the system projected along the x-direction

    Taking our state vector as :math:`Q = (\rho, u, v, p)^T`, the eigenvalues
    are :math:`u - c`, :math:`u`, :math:`u + c`.

    We look at the equations of hydrodynamics in a split fashion --
    i.e., we only consider one dimension at a time.

    Considering advection in the x-direction, the Jacobian matrix for
    the primitive variable formulation of the Euler equations
    projected in the x-direction is::

           / u   0   0 \
           | g   u   0 |
       A = \ 0   0   u /

    The right eigenvectors are::

           /  h  \       /  0  \      /  h  \
      r1 = | -c  |  r2 = |  0  | r3 = |  c  |
           \  0  /       \  1  /      \  0  /

    The left eigenvectors are::

       l1 =     ( 1/(2h),  -h/(2hc),  0 )
       l2 =     ( 0,          0,  1 )
       l3 =     ( -1/(2h), -h/(2hc),  0 )

    The fluxes are going to be defined on the left edge of the
    computational zones::

              |             |             |             |
              |             |             |             |
             -+------+------+------+------+------+------+--
              |     i-1     |      i      |     i+1     |
                           ^ ^           ^
                       q_l,i q_r,i  q_l,i+1

    ``q_r,i`` and ``q_l,i+1`` are computed using the information in zone i,j.

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
    ih, iu, iv, ix : int
        Indices of the height, x-velocity, y-velocity and species in the
        state vector
    nspec : int
        The number of species
    g : float
        Gravitational acceleration
    qv : ndarray
        The primitive state vector
    dqv : ndarray
        Spatial derivitive of the state vector

    Returns
    -------
    out : ndarray, ndarray
        State vector predicted to the left and right edges
    """

    qx, qy, nvar = qv.shape

    q_l = np.zeros_like(qv)
    q_r = np.zeros_like(qv)

    lvec = np.zeros((nvar, nvar))
    rvec = np.zeros((nvar, nvar))
    e_val = np.zeros(nvar)
    betal = np.zeros(nvar)
    betar = np.zeros(nvar)

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    ns = nvar - nspec

    dtdx = dt / dx
    dtdx3 = 0.33333 * dtdx

    # this is the loop over zones.  For zone i, we see q_l[i+1] and q_r[i]
    for i in range(ilo - 2, ihi + 2):
        for j in range(jlo - 2, jhi + 2):

            dq = dqv[i, j, :]
            q = qv[i, j, :]

            cs = np.sqrt(g * q[ih])

            lvec[:, :] = 0.0
            rvec[:, :] = 0.0
            e_val[:] = 0.0

            # compute the eigenvalues and eigenvectors
            if (idir == 1):
                e_val[:ns] = [q[iu] - cs, q[iu], q[iu] + cs]

                lvec[0, :ns] = [cs, -q[ih], 0.0]
                lvec[1, :ns] = [0.0, 0.0, 1.0]
                lvec[2, :ns] = [cs, q[ih], 0.0]

                rvec[0, :ns] = [q[ih], -cs, 0.0]
                rvec[1, :ns] = [0.0, 0.0, 1.0]
                rvec[2, :ns] = [q[ih], cs, 0.0]

                # now the species -- they only have a 1 in their corresponding slot
                e_val[ns:] = q[iu]
                for n in range(ix, ix + nspec):
                    lvec[n, n] = 1.0
                    rvec[n, n] = 1.0

                # multiply by scaling factors
                lvec[0, :] = lvec[0, :] * 0.50 / (cs * q[ih])
                lvec[2, :] = -lvec[2, :] * 0.50 / (cs * q[ih])
            else:
                e_val[:ns] = [q[iv] - cs, q[iv], q[iv] + cs]

                lvec[0, :ns] = [cs, 0.0, -q[ih]]
                lvec[1, :ns] = [0.0, 1.0, 0.0]
                lvec[2, :ns] = [cs, 0.0, q[ih]]

                rvec[0, :ns] = [q[ih], 0.0, -cs]
                rvec[1, :ns] = [0.0, 1.0, 0.0]
                rvec[2, :ns] = [q[ih], 0.0, cs]

                # now the species -- they only have a 1 in their corresponding slot
                e_val[ns:] = q[iv]
                for n in range(ix, ix + nspec):
                    lvec[n, n] = 1.0
                    rvec[n, n] = 1.0

                # multiply by scaling factors
                lvec[0, :] = lvec[0, :] * 0.50 / (cs * q[ih])
                lvec[2, :] = -lvec[2, :] * 0.50 / (cs * q[ih])

            # define the reference states
            if (idir == 1):
                # this is one the right face of the current zone,
                # so the fastest moving eigenvalue is e_val[2] = u + c
                factor = 0.5 * (1.0 - dtdx * max(e_val[2], 0.0))
                q_l[i + 1, j, :] = q + factor * dq

                # left face of the current zone, so the fastest moving
                # eigenvalue is e_val[3] = u - c
                factor = 0.5 * (1.0 + dtdx * min(e_val[0], 0.0))
                q_r[i,  j, :] = q - factor * dq

            else:

                factor = 0.5 * (1.0 - dtdx * max(e_val[2], 0.0))
                q_l[i, j + 1, :] = q + factor * dq

                factor = 0.5 * (1.0 + dtdx * min(e_val[0], 0.0))
                q_r[i, j, :] = q - factor * dq

            # compute the Vhat functions
            for m in range(nvar):
                sum = np.dot(lvec[m, :], dq)

                betal[m] = dtdx3 * (e_val[2] - e_val[m]) * \
                                   (np.copysign(1.0, e_val[m]) + 1.0) * sum
                betar[m] = dtdx3 * (e_val[0] - e_val[m]) * \
                                   (1.0 - np.copysign(1.0, e_val[m])) * sum

            # construct the states
            for m in range(nvar):
                sum_l = np.dot(betal, rvec[:, m])
                sum_r = np.dot(betar, rvec[:, m])

                if (idir == 1):
                    q_l[i + 1, j, m] = q_l[i + 1, j, m] + sum_l
                    q_r[i,  j, m] = q_r[i,  j, m] + sum_r
                else:
                    q_l[i, j + 1, m] = q_l[i, j + 1, m] + sum_l
                    q_r[i, j,  m] = q_r[i, j,  m] + sum_r

    return q_l, q_r


@njit(cache=True)
def riemann_roe(idir, ng,
                ih, ixmom, iymom, ihX, nspec,
                lower_solid, upper_solid,
                g, U_l, U_r):
    r"""
    This is the Roe Riemann solver with entropy fix. The implementation
    follows Toro's SWE book and the clawpack 2d SWE Roe solver.

    Parameters
    ----------
    idir : int
        Are we predicting to the edges in the x-direction (1) or y-direction (2)?
    ng : int
        The number of ghost cells
    ih, ixmom, iymom, ihX : int
        The indices of the height, x-momentum, y-momentum and height*species fractions in the conserved state vector.
    nspec : int
        The number of species
    lower_solid, upper_solid : int
        Are we at lower or upper solid boundaries?
    g : float
        Gravitational acceleration
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
    tol = 0.1e-1  # entropy fix parameter
    # Note that I've basically assumed that cfl = 0.1 here to get away with
    # not passing dx/dt or cfl to this function. If this isn't the case, will need
    # to pass one of these to the function or else: things will go wrong.

    lambda_roe = np.zeros(nvar)
    K_roe = np.zeros((nvar, nvar))
    alpha_roe = np.zeros(nvar)

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny
    ns = nvar - nspec

    for i in range(ilo - 1, ihi + 1):
        for j in range(jlo - 1, jhi + 1):

            # primitive variable states
            h_l = U_l[i, j, ih]

            # un = normal velocity; ut = transverse velocity
            if (idir == 1):
                un_l = U_l[i, j, ixmom] / h_l
            else:
                un_l = U_l[i, j, iymom] / h_l

            h_r = U_r[i, j, ih]

            if (idir == 1):
                un_r = U_r[i, j, ixmom] / h_r
            else:
                un_r = U_r[i, j, iymom] / h_r

            # compute the sound speeds
            c_l = max(smallc, np.sqrt(g * h_l))
            c_r = max(smallc, np.sqrt(g * h_r))

            # Calculate the Roe averages
            U_roe = (U_l[i, j, :] / np.sqrt(h_l) + U_r[i, j, :] / np.sqrt(h_r)) / \
                (np.sqrt(h_l) + np.sqrt(h_r))

            U_roe[ih] = np.sqrt(h_l * h_r)
            c_roe = np.sqrt(0.5 * (c_l**2 + c_r**2))

            delta = U_r[i, j, :] / h_r - U_l[i, j, :] / h_l
            delta[ih] = h_r - h_l

            # e_values and right evectors
            if (idir == 1):
                un_roe = U_roe[ixmom]
            else:
                un_roe = U_roe[iymom]

            K_roe[:, :] = 0.0

            lambda_roe[:3] = np.array([un_roe - c_roe, un_roe, un_roe + c_roe])
            if (idir == 1):
                alpha_roe[:3] = [0.5 * (delta[ih] - U_roe[ih] / c_roe * delta[ixmom]),
                                 U_roe[ih] * delta[iymom],
                                 0.5 * (delta[ih] + U_roe[ih] / c_roe * delta[ixmom])]

                K_roe[0, :3] = [1.0, un_roe - c_roe, U_roe[iymom]]
                K_roe[1, :3] = [0.0, 0.0, 1.0]
                K_roe[2, :3] = [1.0, un_roe + c_roe, U_roe[iymom]]
            else:
                alpha_roe[:3] = [0.5 * (delta[ih] - U_roe[ih] / c_roe * delta[iymom]),
                                 U_roe[ih] * delta[ixmom],
                                 0.5 * (delta[ih] + U_roe[ih] / c_roe * delta[iymom])]

                K_roe[0, :3] = [1.0, U_roe[ixmom], un_roe - c_roe]
                K_roe[1, :3] = [0.0, 1.0, 0.0]
                K_roe[2, :3] = [1.0, U_roe[ixmom], un_roe + c_roe]

            lambda_roe[ns:] = un_roe
            alpha_roe[ns:] = U_roe[ih] * delta[ns:]
            for n in range(ns, nvar):
                K_roe[n, :] = 0.0
                K_roe[n, n] = 1.0

            F[i, j, :] = consFlux(idir, g, ih, ixmom, iymom, ihX, nspec,
                                  U_l[i, j, :])
            F_r = consFlux(idir, g, ih, ixmom, iymom, ihX, nspec,
                           U_r[i, j, :])

            F[i, j, :] = 0.5 * (F[i, j, :] + F_r)

            h_star = 1.0 / g * (0.5 * (c_l + c_r) + 0.25 * (un_l - un_r))**2
            u_star = 0.5 * (un_l + un_r) + c_l - c_r

            c_star = np.sqrt(g * h_star)

            # modified e_values for entropy fix
            if (abs(lambda_roe[0]) < tol):
                lambda_roe[0] = lambda_roe[0] * (u_star - c_star - lambda_roe[0]) / \
                    (u_star - c_star - (un_l - c_l))

            if (abs(lambda_roe[2]) < tol):
                lambda_roe[2] = lambda_roe[2] * (u_star + c_star - lambda_roe[2]) / \
                    (u_star + c_star - (un_r + c_r))

            for n in range(nvar):
                for m in range(nvar):
                    F[i, j, n] -= 0.5 * alpha_roe[m] * \
                        abs(lambda_roe[m]) * K_roe[m, n]

    return F


@njit(cache=True)
def riemann_hllc(idir, ng,
                 ih, ixmom, iymom, ihX, nspec,
                 lower_solid, upper_solid,
                 g, U_l, U_r):
    r"""
    this is the HLLC Riemann solver.  The implementation follows
    directly out of Toro's book.  Note: this does not handle the
    transonic rarefaction.

    Parameters
    ----------
    idir : int
        Are we predicting to the edges in the x-direction (1) or y-direction (2)?
    ng : int
        The number of ghost cells
    ih, ixmom, iymom, ihX : int
        The indices of the height, x-momentum, y-momentum and height*species fractions in the conserved state vector.
    nspec : int
        The number of species
    lower_solid, upper_solid : int
        Are we at lower or upper solid boundaries?
    g : float
        Gravitational acceleration
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
    U_state = np.zeros(nvar)

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    for i in range(ilo - 1, ihi + 1):
        for j in range(jlo - 1, jhi + 1):

            # primitive variable states
            h_l = U_l[i, j, ih]

            # un = normal velocity; ut = transverse velocity
            if (idir == 1):
                un_l = U_l[i, j, ixmom] / h_l
                ut_l = U_l[i, j, iymom] / h_l
            else:
                un_l = U_l[i, j, iymom] / h_l
                ut_l = U_l[i, j, ixmom] / h_l

            h_r = U_r[i, j, ih]

            if (idir == 1):
                un_r = U_r[i, j, ixmom] / h_r
                ut_r = U_r[i, j, iymom] / h_r
            else:
                un_r = U_r[i, j, iymom] / h_r
                ut_r = U_r[i, j, ixmom] / h_r

            # compute the sound speeds
            c_l = max(smallc, np.sqrt(g * h_l))
            c_r = max(smallc, np.sqrt(g * h_r))

            # Estimate the star quantities -- use one of three methods to
            # do this -- the primitive variable Riemann solver, the two
            # shock approximation, or the two rarefaction approximation.
            # Pick the method based on the pressure states at the
            # interface.

            h_avg = 0.5 * (h_l + h_r)
            c_avg = 0.5 * (c_l + c_r)

            hstar = h_avg - 0.25 * (un_r - un_l) * h_avg / c_avg

            # estimate the nonlinear wave speeds

            if (hstar <= h_l):
                # rarefaction
                S_l = un_l - c_l
            else:
                # shock
                S_l = un_l - c_l * np.sqrt(0.5 * (hstar + h_l) * hstar) / h_l

            if (hstar <= h_r):
                # rarefaction
                S_r = un_r + c_r
            else:
                # shock
                S_r = un_r + c_r * np.sqrt(0.5 * (hstar + h_r) * hstar) / h_r

            S_c = (S_l * h_r * (un_r - S_r) - S_r * h_l * (un_l - S_l)) / \
                (h_r * (un_r - S_r) - h_l * (un_l - S_l))

            # figure out which region we are in and compute the state and
            # the interface fluxes using the HLLC Riemann solver
            if (S_r <= 0.0):
                # R region
                U_state[:] = U_r[i, j, :]

                F[i, j, :] = consFlux(idir, g, ih, ixmom, iymom, ihX, nspec,
                                      U_state)

            elif (S_r > 0.0 and S_c <= 0):
                # R* region
                HLLCfactor = h_r * (S_r - un_r) / (S_r - S_c)

                U_state[ih] = HLLCfactor

                if (idir == 1):
                    U_state[ixmom] = HLLCfactor * S_c
                    U_state[iymom] = HLLCfactor * ut_r
                else:
                    U_state[ixmom] = HLLCfactor * ut_r
                    U_state[iymom] = HLLCfactor * S_c

                # species
                if (nspec > 0):
                    U_state[ihX:ihX + nspec] = HLLCfactor * \
                        U_r[i, j, ihX:ihX + nspec] / h_r

                # find the flux on the right interface
                F[i, j, :] = consFlux(idir, g, ih, ixmom, iymom, ihX, nspec,
                                      U_r[i, j, :])

                # correct the flux
                F[i, j, :] = F[i, j, :] + S_r * (U_state - U_r[i, j, :])

            elif (S_c > 0.0 and S_l < 0.0):
                # L* region
                HLLCfactor = h_l * (S_l - un_l) / (S_l - S_c)

                U_state[ih] = HLLCfactor

                if (idir == 1):
                    U_state[ixmom] = HLLCfactor * S_c
                    U_state[iymom] = HLLCfactor * ut_l
                else:
                    U_state[ixmom] = HLLCfactor * ut_l
                    U_state[iymom] = HLLCfactor * S_c

                # species
                if (nspec > 0):
                    U_state[ihX:ihX + nspec] = HLLCfactor * \
                        U_l[i, j, ihX:ihX + nspec] / h_l

                # find the flux on the left interface
                F[i, j, :] = consFlux(idir, g, ih, ixmom, iymom, ihX, nspec,
                                      U_l[i, j, :])

                # correct the flux
                F[i, j, :] = F[i, j, :] + S_l * (U_state - U_l[i, j, :])

            else:
                # L region
                U_state[:] = U_l[i, j, :]

                F[i, j, :] = consFlux(idir, g, ih, ixmom, iymom, ihX, nspec,
                                      U_state)
    return F


@njit(cache=True)
def consFlux(idir, g, ih, ixmom, iymom, ihX, nspec, U_state):
    r"""
    Calculate the conserved flux for the shallow water equations. In the
    x-direction, this is given by::

            /      hu       \
        F = | hu^2 + gh^2/2 |
            \      huv      /

    Parameters
    ----------
    idir : int
        Are we predicting to the edges in the x-direction (1) or y-direction (2)?
    g : float
        Graviational acceleration
    ih, ixmom, iymom, ihX : int
        The indices of the height, x-momentum, y-momentum, height*species fraction in the conserved state vector.
    nspec : int
        The number of species
    U_state : ndarray
        Conserved state vector.

    Returns
    -------
    out : ndarray
        Conserved flux
    """

    F = np.zeros_like(U_state)

    u = U_state[ixmom] / U_state[ih]
    v = U_state[iymom] / U_state[ih]

    if (idir == 1):
        F[ih] = U_state[ih] * u
        F[ixmom] = U_state[ixmom] * u + 0.5 * g * U_state[ih]**2
        F[iymom] = U_state[iymom] * u
        if (nspec > 0):
            F[ihX:ihX + nspec] = U_state[ihX:ihX + nspec] * u

    else:
        F[ih] = U_state[ih] * v
        F[ixmom] = U_state[ixmom] * v
        F[iymom] = U_state[iymom] * v + 0.5 * g * U_state[ih]**2
        if (nspec > 0):
            F[ihX:ihX + nspec] = U_state[ihX:ihX + nspec] * v

    return F
