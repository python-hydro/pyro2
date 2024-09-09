import numpy as np
from numba import njit


@njit(cache=True)
def states(idir, ng, dx, dt,
           irho, iu, iv, ip, ix, nspec,
           gamma, qv, dqv):
    r"""
    predict the cell-centered state to the edges in one-dimension
    using the reconstructed, limited slopes.

    We follow the convection here that ``V_l[i]`` is the left state at the
    i-1/2 interface and ``V_l[i+1]`` is the left state at the i+1/2
    interface.

    We need the left and right eigenvectors and the eigenvalues for
    the system projected along the x-direction.

    Taking our state vector as :math:`Q = (\rho, u, v, p)^T`, the eigenvalues
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
    dx : ndarray
        The cell spacing
    dt : float
        The timestep
    irho, iu, iv, ip, ix : int
        Indices of the density, x-velocity, y-velocity, pressure and species in the
        state vector
    nspec : int
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

    ns = nvar - nspec

    dtdx = dt / dx
    dtdx4 = 0.25 * dtdx

    lvec = np.zeros((nvar, nvar))
    rvec = np.zeros((nvar, nvar))
    e_val = np.zeros(nvar)
    betal = np.zeros(nvar)
    betar = np.zeros(nvar)

    # this is the loop over zones.  For zone i, we see q_l[i+1] and q_r[i]
    for i in range(ilo - 2, ihi + 2):
        for j in range(jlo - 2, jhi + 2):

            dq = dqv[i, j, :]
            q = qv[i, j, :]

            cs = np.sqrt(gamma * q[ip] / q[irho])

            lvec[:, :] = 0.0
            rvec[:, :] = 0.0
            e_val[:] = 0.0

            # compute the eigenvalues and eigenvectors
            if idir == 1:
                e_val[:] = np.array([q[iu] - cs, q[iu], q[iu], q[iu] + cs])

                lvec[0, :ns] = [0.0, -0.5 *
                                 q[irho] / cs, 0.0, 0.5 / (cs * cs)]
                lvec[1, :ns] = [1.0, 0.0,
                                 0.0, -1.0 / (cs * cs)]
                lvec[2, :ns] = [0.0, 0.0, 1.0, 0.0]
                lvec[3, :ns] = [0.0, 0.5 *
                                 q[irho] / cs,  0.0, 0.5 / (cs * cs)]

                rvec[0, :ns] = [1.0, -cs / q[irho], 0.0, cs * cs]
                rvec[1, :ns] = [1.0, 0.0, 0.0, 0.0]
                rvec[2, :ns] = [0.0, 0.0, 1.0, 0.0]
                rvec[3, :ns] = [1.0, cs / q[irho],  0.0, cs * cs]

                # now the species -- they only have a 1 in their corresponding slot
                e_val[ns:] = q[iu]
                for n in range(ix, ix + nspec):
                    lvec[n, n] = 1.0
                    rvec[n, n] = 1.0

            else:
                e_val[:] = np.array([q[iv] - cs, q[iv], q[iv], q[iv] + cs])

                lvec[0, :ns] = [0.0, 0.0, -0.5 *
                                 q[irho] / cs, 0.5 / (cs * cs)]
                lvec[1, :ns] = [1.0, 0.0,
                                 0.0,             -1.0 / (cs * cs)]
                lvec[2, :ns] = [0.0, 1.0, 0.0,             0.0]
                lvec[3, :ns] = [0.0, 0.0, 0.5 *
                                 q[irho] / cs,  0.5 / (cs * cs)]

                rvec[0, :ns] = [1.0, 0.0, -cs / q[irho], cs * cs]
                rvec[1, :ns] = [1.0, 0.0, 0.0,       0.0]
                rvec[2, :ns] = [0.0, 1.0, 0.0,       0.0]
                rvec[3, :ns] = [1.0, 0.0, cs / q[irho],  cs * cs]

                # now the species -- they only have a 1 in their corresponding slot
                e_val[ns:] = q[iv]
                for n in range(ix, ix + nspec):
                    lvec[n, n] = 1.0
                    rvec[n, n] = 1.0

            # define the reference states
            if idir == 1:
                # this is one the right face of the current zone,
                # so the fastest moving eigenvalue is e_val[3] = u + c
                factor = 0.5 * (1.0 - dtdx[i, j] * max(e_val[3], 0.0))
                q_l[i + 1, j, :] = q + factor * dq

                # left face of the current zone, so the fastest moving
                # eigenvalue is e_val[3] = u - c
                factor = 0.5 * (1.0 + dtdx[i, j] * min(e_val[0], 0.0))
                q_r[i,  j, :] = q - factor * dq

            else:

                factor = 0.5 * (1.0 - dtdx[i, j] * max(e_val[3], 0.0))
                q_l[i, j + 1, :] = q + factor * dq

                factor = 0.5 * (1.0 + dtdx[i, j] * min(e_val[0], 0.0))
                q_r[i, j, :] = q - factor * dq

            # compute the Vhat functions
            for m in range(nvar):
                asum = np.dot(lvec[m, :], dq)

                # Should we change to max(e_val[3], 0.0) and min(e_val[0], 0.0)?
                betal[m] = dtdx4[i, j] * (e_val[3] - e_val[m]) * \
                    (np.copysign(1.0, e_val[m]) + 1.0) * asum
                betar[m] = dtdx4[i, j] * (e_val[0] - e_val[m]) * \
                    (1.0 - np.copysign(1.0, e_val[m])) * asum

            # construct the states
            for m in range(nvar):
                sum_l = np.dot(betal, np.ascontiguousarray(rvec[:, m]))
                sum_r = np.dot(betar, np.ascontiguousarray(rvec[:, m]))

                if idir == 1:
                    q_l[i + 1, j, m] = q_l[i + 1, j, m] + sum_l
                    q_r[i,  j, m] = q_r[i,  j, m] + sum_r
                else:
                    q_l[i, j + 1, m] = q_l[i, j + 1, m] + sum_l
                    q_r[i, j,  m] = q_r[i, j,  m] + sum_r

    return q_l, q_r


@njit(cache=True)
def artificial_viscosity(ng, dx, dy, Lx, Ly,
                         xmin, ymin, coord_type,
                         cvisc, u, v):
    r"""
    Compute the artificial viscosity.  Here, we compute edge-centered
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
    xmin, ymin : float
        Min physical x, y boundary
    Lx, Ly: ndarray
        Cell size in x, y direction
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
    divU = np.zeros((qx, qy))

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    # Let's first compute divergence at the vertex
    # First compute the left and right x-velocities by
    # averaging over the y-interface.
    # As well as the top and bottom y-velocities by
    # averaging over the x-interface.
    # Then a simple difference is done between the right and left,
    # and top and bottom to get the divergence at the vertex.

    for i in range(ilo - 1, ihi + 1):
        for j in range(jlo - 1, jhi + 1):

            # For Cartesian2d:
            if coord_type == 0:
                # Find the average right and left u velocity
                ur = 0.5 * (u[i, j] + u[i, j - 1])
                ul = 0.5 * (u[i - 1, j] + u[i - 1, j - 1])

                # Find the average top and bottom v velocity
                vt = 0.5 * (v[i, j] + v[i - 1, j])
                vb = 0.5 * (v[i, j - 1] + v[i - 1, j - 1])

                # Finite difference to get ux and vy
                ux = (ur - ul) / dx
                vy = (vt - vb) / dy

                # Find div(U)_{i-1/2, j-1/2}
                divU[i, j] = ux + vy

            # For SphericalPolar:
            else:
                # cell-centered r-coord of right, left cell and face-centered r
                rr = (i + 1 - ng) * dx + xmin
                rl = (i - ng) * dx + xmin
                rc = 0.5 * (rr + rl)

                # cell-centered sin(theta) of top, bot cell and face-centered
                sint = np.sin((j + 1 - ng) * dy + ymin)
                sinb = np.sin((j - ng) * dy + ymin)
                sinc = np.sin((j + 0.5 - ng) * dy + ymin)

                # Find the average right and left u velocity
                ur = 0.5 * (u[i, j] + u[i, j - 1])
                ul = 0.5 * (u[i - 1, j] + u[i - 1, j - 1])

                # Find the average top and bottom v velocity
                vt = 0.5 * (v[i, j] + v[i - 1, j])
                vb = 0.5 * (v[i, j - 1] + v[i - 1, j - 1])

                # Finite difference to get ux and vy
                ux = (ur * rr * rr - ul * rl * rl) / (rc * rc * dx)
                vy = (sint * vt - sinb * vb) / (rc * sinc * dy)

                # Find div(U)_{i-1/2, j-1/2}
                divU[i, j] = ux + vy

    # Compute divergence at the face by averaging over divergence at vertex
    for i in range(ilo, ihi):
        for j in range(jlo, jhi):

            divU_x = 0.5 * (divU[i, j] + divU[i, j + 1])
            divU_y = 0.5 * (divU[i, j] + divU[i + 1, j])

            avisco_x[i, j] = cvisc * max(-divU_x * Lx[i, j], 0.0)
            avisco_y[i, j] = cvisc * max(-divU_y * Ly[i, j], 0.0)

    return avisco_x, avisco_y
