import numpy as np
from numba import njit


@njit(cache=True)
def states(idir, ng, dx, dt,
           ivars,
           gamma, qv, dqv):
    r"""
    predict the cell-centered state to the edges in one-dimension
    using the reconstructed, limited slopes.

    We follow the convection here that ``V_l[i]`` is the left state at the
    i-1/2 interface and ``V_l[i+1]`` is the left state at the i+1/2
    interface.

    We need the left and right eigenvectors and the eigenvalues for
    the system projected along the x-direction.

    Taking our state vector as :math:`Q = (\rho, u, v, p, bx, by)^T`, the eigenvalues
    are :math:`u - c`, :math:`u`, :math:`u + c`.

    We look at the equations of hydrodynamics in a split fashion --
    i.e., we only consider one dimension at a time.

    Considering advection in the x-direction, the Jacobian matrix for
    the primitive variable formulation of the Euler equations
    projected in the x-direction is::

             / u   r   0   0 \
             | 0   u   0  1/r |
         A = | 0   0   u   0  |
             \ 0  rc^2 0   u  /

    The right eigenvectors are::

             /  1  \        / 1 \        / 0 \        /  1  \
             |-c/r |        | 0 |        | 0 |        | c/r |
        r1 = |  0  |   r2 = | 0 |   r3 = | 1 |   r4 = |  0  |
             \ c^2 /        \ 0 /        \ 0 /        \ c^2 /

    In particular, we see from r3 that the transverse velocity (v in
    this case) is simply advected at a speed u in the x-direction.

    The left eigenvectors are::

         l1 =     ( 0,  -r/(2c),  0, 1/(2c^2) )
         l2 =     ( 1,     0,     0,  -1/c^2  )
         l3 =     ( 0,     0,     1,     0    )
         l4 =     ( 0,   r/(2c),  0, 1/(2c^2) )

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
    ivars.irho, ivars.iu, ivars.iv, ivars.ip, ix : int
        Indices of the density, x-velocity, y-velocity, pressure and species in the
        state vector
    ivars.naux : int
        The number of species
    gamma : float
        Adiabatic index
    qv : ndarray
        The primitive state vector
    dqv : ndarray
        Spatial derivative of the state vector

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

    # this is the loop over zones.  For zone i, we see q_l[i+1] and q_r[i]
    for i in range(ilo - 2, ihi + 2):
        for j in range(jlo - 2, jhi + 2):

            dq = dqv[i, j, :]
            q = qv[i, j, :]

            if (idir == 1):
                q_l[i + 1, j, :] = q + 0.5 * dq
                q_r[i,  j, :] = q - 0.5 * dq

            else:
                q_l[i, j + 1, :] = q + 0.5 * dq
                q_r[i, j, :] = q - 0.5 * dq

    return q_l, q_r


@njit(cache=True)
def riemann_adiabatic(idir, ng,
                      ivars,
                      lower_solid, upper_solid,
                      gamma, U_l, U_r):

    qx, qy, nvar = U_l.shape

    F = np.zeros((qx, qy, nvar))

    smallc = 1.e-10
    # smallrho = 1.e-10
    smallp = 1.e-10

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    for i in range(ilo - 1, ihi + 1):
        for j in range(jlo - 1, jhi + 1):
            # primitive variable states
            rho_l = U_l[i, j, ivars.idens]

            # un = normal velocity; ut = transverse velocity
            if (idir == 1):
                un_l = U_l[i, j, ivars.ixmom] / rho_l
                ut_l = U_l[i, j, ivars.iymom] / rho_l
            else:
                un_l = U_l[i, j, ivars.iymom] / rho_l
                ut_l = U_l[i, j, ivars.ixmom] / rho_l

            rhoe_l = U_l[i, j, ivars.iener] - 0.5 * rho_l * (un_l**2 + ut_l**2)

            p_l = rhoe_l * (gamma - 1.0)
            p_l = max(p_l, smallp)

            rho_r = U_r[i, j, ivars.idens]

            if (idir == 1):
                un_r = U_r[i, j, ivars.ixmom] / rho_r
                ut_r = U_r[i, j, ivars.iymom] / rho_r
            else:
                un_r = U_r[i, j, ivars.iymom] / rho_r
                ut_r = U_r[i, j, ivars.ixmom] / rho_r

            rhoe_r = U_r[i, j, ivars.iener] - 0.5 * rho_r * (un_r**2 + ut_r**2)

            p_r = rhoe_r * (gamma - 1.0)
            p_r = max(p_r, smallp)

            bx_l = U_l[i, j, ivars.ixmag]
            by_l = U_l[i, j, ivars.iymag]
            bx_r = U_r[i, j, ivars.ixmag]
            by_r = U_r[i, j, ivars.iymag]

            # and the regular sound speeds
            c_l = max(smallc, np.sqrt(gamma * p_l / rho_l))
            c_r = max(smallc, np.sqrt(gamma * p_r / rho_r))

            # find alven wavespeeds and fast and slow magnetosonic wavespeeds
            cA2_l = (bx_l**2 + by_l**2) / rho_l
            cA2_r = (bx_r**2 + by_r**2) / rho_r

            if idir == 1:
                cAx2_l = bx_l**2 / rho_l
                cAx2_r = bx_r**2 / rho_r
            else:
                cAx2_l = by_l**2 / rho_l
                cAx2_r = by_r**2 / rho_r

            cf_l = np.sqrt(
                0.5 * (c_l**2 + cA2_l + np.sqrt((c_l**2 + cA2_l)**2 - 4 * c_l**2 * cAx2_l)))
            cf_r = np.sqrt(
                0.5 * (c_r**2 + cA2_r + np.sqrt((c_r**2 + cA2_r)**2 - 4 * c_r**2 * cAx2_r)))

            cs_l = np.sqrt(
                0.5 * (c_l**2 + cA2_l - np.sqrt((c_l**2 + cA2_l)**2 - 4 * c_l**2 * cAx2_l)))
            cs_r = np.sqrt(
                0.5 * (c_r**2 + cA2_r - np.sqrt((c_r**2 + cA2_r)**2 - 4 * c_r**2 * cAx2_r)))

            evals_l = (un_l - cf_l, un_l - np.sqrt(cAx2_l), un_l -
                       cs_l, un_l + cs_l, un_l + np.sqrt(cAx2_l), un_l + cf_l)
            evals_r = (un_r - cf_r, un_r - np.sqrt(cAx2_r), un_r -
                       cs_r, un_r + cs_r, un_r + np.sqrt(cAx2_r), un_r + cf_r)

            bp = np.max(np.max(np.max(evals_r), un_r + cf_r), 0)
            bm = np.min(np.min(np.min(evals_l), un_l - cf_l), 0)

            f_l = consFlux(idir, gamma, ivars, U_l[i, j, :])
            f_r = consFlux(idir, gamma, ivars, U_r[i, j, :])

            F[i, j, :] = (bp * f_l - bm * f_r) / (bp - bm) + bp * \
                bm / (bp - bm) * (U_r[i, j, :] - U_l[i, j, :])

    return F


@njit(cache=True)
def consFlux(idir, gamma, ivars, U_state):
    r"""
    Calculate the conservative flux.

    Parameters
    ----------
    idir : int
        Are we predicting to the edges in the x-direction (1) or y-direction (2)?
    gamma : float
        Adiabatic index
    ivars.idens, ivars.ixmom, ivars.iymom, ivars.iener, ivars.irhoX : int
        The indices of the density, x-momentum, y-momentum, internal energy density
        and species partial densities in the conserved state vector.
    ivars.naux : int
        The number of species
    U_state : ndarray
        Conserved state vector.

    Returns
    -------
    out : ndarray
        Conserved flux
    """

    F = np.zeros_like(U_state)

    u = U_state[ivars.ixmom] / U_state[ivars.idens]
    v = U_state[ivars.iymom] / U_state[ivars.idens]

    p = (U_state[ivars.iener] - 0.5 * U_state[ivars.idens]
         * (u * u + v * v)) * (gamma - 1.0)

    bx = U_state[ivars.ixmag]
    by = U_state[ivars.iymag]

    if (idir == 1):
        F[ivars.idens] = U_state[ivars.idens] * u
        F[ivars.ixmom] = U_state[ivars.ixmom] * \
            u + p + (bx**2 + by**2) * 0.5 - bx**2
        F[ivars.iymom] = U_state[ivars.iymom] * u - bx * by
        F[ivars.iener] = (U_state[ivars.iener] + p +
                          (bx**2 + by**2) * 0.5) * u - bx * (bx * u + by * v)
        F[ivars.ixmag] = 0
        F[ivars.iymag] = by * u - bx * v

        if (ivars.naux > 0):
            F[ivars.irhoX:ivars.irhoX +
                ivars.naux] = U_state[ivars.irhoX:ivars.irhoX + ivars.naux] * u

    else:
        F[ivars.idens] = U_state[ivars.idens] * v
        F[ivars.ixmom] = U_state[ivars.ixmom] * v - bx * by
        F[ivars.iymom] = U_state[ivars.iymom] * \
            v + p + (bx**2 + by**2) * 0.5 - by**2
        F[ivars.iener] = (U_state[ivars.iener] + p +
                          (bx**2 + by**2) * 0.5) * v - by * (bx * u + by * v)
        F[ivars.ixmag] = bx * v - by * u
        F[ivars.iymag] = 0

        if (ivars.naux > 0):
            F[ivars.irhoX:ivars.irhoX +
                ivars.naux] = U_state[ivars.irhoX:ivars.irhoX + ivars.naux] * v

    return F


@njit(cache=True)
def emf(ng, ivars, dx, dy, U, Ux, Uy):
    r"""
    Calculate the EMF at cell corners
    """

    qx, qy, nvar = U.shape

    E = np.zeros((qx, qy))
    Ex = np.zeros((qx, qy))  # x-edges
    Ey = np.zeros((qx, qy))  # y-edges

    Ec = np.zeros((qx, qy))  # corner

    dEdy_14 = np.zeros((qx, qy))
    dEdx_14 = np.zeros((qx, qy))

    dEdy_34 = np.zeros((qx, qy))
    dEdx_34 = np.zeros((qx, qy))

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    for i in range(ilo - 2, ihi + 2):
        for j in range(jlo - 2, jhi + 2):

            u = U[i, j, ivars.ixmom] / U[i, j, ivars.idens]
            v = U[i, j, ivars.iymom] / U[i, j, ivars.idens]
            bx = U[i, j, ivars.ixmag]
            by = U[i, j, ivars.iymag]

            E[i, j] = -(u * by - v * bx)

            u = Ux[i, j, ivars.ixmom] / Ux[i, j, ivars.idens]
            v = Ux[i, j, ivars.iymom] / Ux[i, j, ivars.idens]
            bx = Ux[i, j, ivars.ixmag]
            by = Ux[i, j, ivars.iymag]

            Ex[i, j] = -(u * by - v * bx)

            u = Uy[i, j, ivars.ixmom] / Uy[i, j, ivars.idens]
            v = Uy[i, j, ivars.iymom] / Uy[i, j, ivars.idens]
            bx = Uy[i, j, ivars.ixmag]
            by = Uy[i, j, ivars.iymag]

            Ey[i, j] = -(u * by - v * bx)

    for i in range(ilo - 1, ihi + 1):
        for j in range(jlo - 1, jhi + 1):

            # get the -1/4 states
            dEdy_14[i, j] = 2 * (E[i, j] - Ey[i, j]) / dy

            dEdx_14[i, j] = 2 * (E[i, j] - Ex[i, j]) / dx

            # get the -3/4 states
            dEdy_34[i, j] = 2 * (Ey[i, j] - E[i, j - 1]) / dy

            dEdx_34[i, j] = 2 * (Ey[i, j] - Ex[i - 1, j]) / dx

            # now get the corner states
            u = Ux[i, j, ivars.ixmom] / Ux[i, j, ivars.idens]
            if u > 0:
                dEdyx_14 = dEdy_14[i - 1, j]
                dEdyx_34 = dEdy_34[i - 1, j]
            elif u < 0:
                dEdyx_14 = dEdy_14[i, j]
                dEdyx_34 = dEdy_34[i, j]
            else:
                dEdyx_14 = 0.5 * (dEdy_14[i - 1, j] + dEdy_14[i, j])
                dEdyx_34 = 0.5 * (dEdy_34[i - 1, j] + dEdy_34[i, j])

            v = Uy[i, j, ivars.iymom] / Uy[i, j, ivars.idens]
            if u > 0:
                dEdxy_14 = dEdx_14[i, j - 1]
                dEdxy_34 = dEdx_34[i, j - 1]
            elif u < 0:
                dEdxy_14 = dEdx_14[i, j]
                dEdxy_34 = dEdx_34[i, j]
            else:
                dEdxy_14 = 0.5 * (dEdx_14[i, j - 1] + dEdx_14[i, j])
                dEdxy_34 = 0.5 * (dEdx_34[i, j - 1] + dEdx_34[i, j])

            Ec[i, j] = 0.25 * (Ex[i, j] + Ex[i, j + 1] + Ey[i, j] + Ey[i + 1, j]) + \
                0.125 / dx * (dEdyx_14 - dEdyx_34) + 0.125 / dx * (dEdxy_14 - dEdxy_34)

    return Ec

@njit(cache=True)
def sources(idir, ng, ivars, dx, U, Ux):
    r"""
    Calculate source terms on the idir-interface. U is the cell-centered state,
    Ux should be a state on the idir-interface.

    Assume Bz = vz = 0.
    """
    qx, qy, nvar = U.shape

    S = np.zeros((qx, qy))

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    for i in range(ilo - 1, ihi + 1):
        for j in range(jlo - 1, jhi + 1):
            if idir == 1:
                S[i,j,ivars.ixmom] = U[i,j,ivars.ixmag] * (Ux[i+1,j,ivars.ixmag] - Ux[i,j,ivars.ixmag]) / dx
                S[i,j,ivars.iymom] = U[i,j,ivars.iymag] * (Ux[i+1,j,ivars.ixmag] - Ux[i,j,ivars.ixmag]) / dx
            else:
                S[i,j,ivars.ixmom] = U[i,j,ivars.ixmag] * (Ux[i,j+1,ivars.ixmag] - Ux[i,j,ivars.ixmag]) / dx
                S[i,j,ivars.iymom] = U[i,j,ivars.iymag] * (Ux[i,j+1,ivars.ixmag] - Ux[i,j,ivars.ixmag]) / dx

    return S

@njit(cache=True)
def artificial_viscosity(ng, dx, dy,
                         cvisc, u, v):
    r"""
    Compute the artifical viscosity.  Here, we compute edge-centered
    approximations to the divergence of the velocity.  This follows
    directly Colella \ Woodward (1984) Eq. 4.5

    data locations::

        j+3/2--+---------+---------+---------+
               |         |         |         |
          j+1  +         |         |         |
               |         |         |         |
        j+1/2--+---------+---------+---------+
               |         |         |         |
             j +         X         |         |
               |         |         |         |
        j-1/2--+---------+----Y----+---------+
               |         |         |         |
           j-1 +         |         |         |
               |         |         |         |
        j-3/2--+---------+---------+---------+
               |    |    |    |    |    |    |
                   i-1        i        i+1
             i-3/2     i-1/2     i+1/2     i+3/2

    ``X`` is the location of ``avisco_x[i,j]``
    ``Y`` is the location of ``avisco_y[i,j]``

    Parameters
    ----------
    ng : int
        The number of ghost cells
    dx, dy : float
        Cell spacings
    cvisc : float
        viscosity parameter
    u, v : ndarray
        x- and y-velocities

    Returns
    -------
    out : ndarray, ndarray
        Artificial viscosity in the x- and y-directions
    """

    qx, qy = u.shape

    avisco_x = np.zeros((qx, qy))
    avisco_y = np.zeros((qx, qy))

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    for i in range(ilo - 1, ihi + 1):
        for j in range(jlo - 1, jhi + 1):

                # start by computing the divergence on the x-interface.  The
                # x-difference is simply the difference of the cell-centered
                # x-velocities on either side of the x-interface.  For the
                # y-difference, first average the four cells to the node on
                # each end of the edge, and: difference these to find the
                # edge centered y difference.
            divU_x = (u[i, j] - u[i - 1, j]) / dx + \
                0.25 * (v[i, j + 1] + v[i - 1, j + 1] -
                        v[i, j - 1] - v[i - 1, j - 1]) / dy

            avisco_x[i, j] = cvisc * max(-divU_x * dx, 0.0)

            # now the y-interface value
            divU_y = 0.25 * (u[i + 1, j] + u[i + 1, j - 1] - u[i - 1, j] - u[i - 1, j - 1]) / dx + \
                (v[i, j] - v[i, j - 1]) / dy

            avisco_y[i, j] = cvisc * max(-divU_y * dy, 0.0)

    return avisco_x, avisco_y
