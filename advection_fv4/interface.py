import numpy as np
from numba import njit


@njit(cache=True)
def states(a, ng, idir):
    r"""
    Predict the cell-centered state to the edges in one-dimension using the
    reconstructed, limited slopes. We use a fourth-order Godunov method.

    Our convention here is that:

        ``al[i,j]``   will be :math:`al_{i-1/2,j}`,

        ``al[i+1,j]`` will be :math:`al_{i+1/2,j}`.

    Parameters
    ----------
    a : ndarray
        The cell-centered state.
    ng : int
        The number of ghost cells
    idir : int
        Are we predicting to the edges in the x-direction (1) or y-direction (2)?

    Returns
    -------
    out : ndarray, ndarray
        The state predicted to the left and right edges.
    """

    qx, qy = a.shape

    al = np.zeros((qx, qy))
    ar = np.zeros((qx, qy))

    a_int = np.zeros((qx, qy))
    dafm = np.zeros((qx, qy))
    dafp = np.zeros((qx, qy))
    d2af = np.zeros((qx, qy))
    d2ac = np.zeros((qx, qy))
    d3a = np.zeros((qx, qy))

    C2 = 1.25
    C3 = 0.1

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    # we need interface values on all faces of the domain
    if (idir == 1):

        for i in range(ilo - 2, ihi + 3):
            for j in range(jlo - 1, jhi + 1):

                # interpolate to the edges
                a_int[i, j] = (7.0 / 12.0) * (a[i - 1, j] + a[i, j]) - \
                    (1.0 / 12.0) * (a[i - 2, j] + a[i + 1, j])

                al[i, j] = a_int[i, j]
                ar[i, j] = a_int[i, j]

        for i in range(ilo - 2, ihi + 3):
            for j in range(jlo - 1, jhi + 1):
                # these live on cell-centers
                dafm[i, j] = a[i, j] - a_int[i, j]
                dafp[i, j] = a_int[i + 1, j] - a[i, j]

                # these live on cell-centers
                d2af[i, j] = 6.0 * (a_int[i, j] - 2.0 *
                                    a[i, j] + a_int[i + 1, j])

        for i in range(ilo - 3, ihi + 3):
            for j in range(jlo - 1, jhi + 1):
                d2ac[i, j] = a[i - 1, j] - 2.0 * a[i, j] + a[i + 1, j]

        for i in range(ilo - 2, ihi + 3):
            for j in range(jlo - 1, jhi + 1):
                # this lives on the interface
                d3a[i, j] = d2ac[i, j] - d2ac[i - 1, j]

        # this is a look over cell centers, affecting
        # i-1/2,R and i+1/2,L
        for i in range(ilo - 1, ihi + 1):
            for j in range(jlo - 1, jhi + 1):

                # limit? MC Eq. 24 and 25
                if (dafm[i, j] * dafp[i, j] <= 0.0 or
                        (a[i, j] - a[i - 2, j]) * (a[i + 2, j] - a[i, j]) <= 0.0):

                    # we are at an extrema

                    s = np.copysign(1.0, d2ac[i, j])
                    if (s == np.copysign(1.0, d2ac[i - 1, j]) and s == np.copysign(1.0, d2ac[i + 1, j]) and
                            s == np.copysign(1.0, d2af[i, j])):
                        # MC Eq. 26
                        d2a_lim = s * min(abs(d2af[i, j]), C2 * abs(d2ac[i - 1, j]),
                                          C2 * abs(d2ac[i, j]), C2 * abs(d2ac[i + 1, j]))
                    else:
                        d2a_lim = 0.0

                    if (abs(d2af[i, j]) <= 1.e-12 * max(abs(a[i - 2, j]), abs(a[i - 1, j]),
                                                        abs(a[i, j]), abs(a[i + 1, j]), abs(a[i + 2, j]))):
                        rho = 0.0
                    else:
                        # MC Eq. 27
                        rho = d2a_lim / d2af[i, j]

                    if (rho < 1.0 - 1.e-12):
                        # we may need to limit -- these quantities are at cell-centers
                        d3a_min = min(d3a[i - 1, j], d3a[i, j],
                                      d3a[i + 1, j], d3a[i + 2, j])
                        d3a_max = max(d3a[i - 1, j], d3a[i, j],
                                      d3a[i + 1, j], d3a[i + 2, j])

                        if (C3 * max(abs(d3a_min), abs(d3a_max)) <= (d3a_max - d3a_min)):
                            # limit
                            if (dafm[i, j] * dafp[i, j] < 0.0):
                                # Eqs. 29, 30
                                ar[i, j] = a[i, j] - rho * \
                                    dafm[i, j]  # note: typo in Eq 29
                                al[i + 1, j] = a[i, j] + rho * dafp[i, j]
                            elif (abs(dafm[i, j]) >= 2.0 * abs(dafp[i, j])):
                                # Eq. 31
                                ar[i, j] = a[i, j] - 2.0 * \
                                    (1.0 - rho) * dafp[i, j] - rho * dafm[i, j]
                            elif (abs(dafp[i, j]) >= 2.0 * abs(dafm[i, j])):
                                # Eq. 32
                                al[i + 1, j] = a[i, j] + 2.0 * \
                                    (1.0 - rho) * dafm[i, j] + rho * dafp[i, j]

                else:
                    # if Eqs. 24 or 25 didn't hold we still may need to limit
                    if (abs(dafm[i, j]) >= 2.0 * abs(dafp[i, j])):
                        ar[i, j] = a[i, j] - 2.0 * dafp[i, j]

                    if (abs(dafp[i, j]) >= 2.0 * abs(dafm[i, j])):
                        al[i + 1, j] = a[i, j] + 2.0 * dafm[i, j]

    elif (idir == 2):

        for i in range(ilo - 1, ihi + 1):
            for j in range(jlo - 2, jhi + 3):

                # interpolate to the edges
                a_int[i, j] = (7.0 / 12.0) * (a[i, j - 1] + a[i, j]) - \
                    (1.0 / 12.0) * (a[i, j - 2] + a[i, j + 1])

                al[i, j] = a_int[i, j]
                ar[i, j] = a_int[i, j]

        for i in range(ilo - 1, ihi + 1):
            for j in range(jlo - 2, jhi + 3):
                # these live on cell-centers
                dafm[i, j] = a[i, j] - a_int[i, j]
                dafp[i, j] = a_int[i, j + 1] - a[i, j]

                # these live on cell-centers
                d2af[i, j] = 6.0 * (a_int[i, j] - 2.0 *
                                    a[i, j] + a_int[i, j + 1])

        for i in range(ilo - 1, ihi + 1):
            for j in range(jlo - 3, jhi + 3):
                d2ac[i, j] = a[i, j - 1] - 2.0 * a[i, j] + a[i, j + 1]

        for i in range(ilo - 1, ihi + 1):
            for j in range(jlo - 2, jhi + 2):
                # this lives on the interface
                d3a[i, j] = d2ac[i, j] - d2ac[i, j - 1]

        # this is a look over cell centers, affecting
        # j-1/2,R and j+1/2,L
        for i in range(ilo - 1, ihi + 1):
            for j in range(jlo - 1, jhi + 1):

                # limit? MC Eq. 24 and 25
                if (dafm[i, j] * dafp[i, j] <= 0.0 or
                        (a[i, j] - a[i, j - 2]) * (a[i, j + 2] - a[i, j]) <= 0.0):

                    # we are at an extrema

                    s = np.copysign(1.0, d2ac[i, j])
                    if (s == np.copysign(1.0, d2ac[i, j - 1]) and s == np.copysign(1.0, d2ac[i, j + 1]) and
                            s == np.copysign(1.0, d2af[i, j])):
                        # MC Eq. 26
                        d2a_lim = s * min(abs(d2af[i, j]), C2 * abs(d2ac[i, j - 1]),
                                          C2 * abs(d2ac[i, j]), C2 * abs(d2ac[i, j + 1]))
                    else:
                        d2a_lim = 0.0

                    if (abs(d2af[i, j]) <= 1.e-12 * max(abs(a[i, j - 2]), abs(a[i, j - 1]),
                                                        abs(a[i, j]), abs(a[i, j + 1]), abs(a[i, j + 2]))):
                        rho = 0.0
                    else:
                        # MC Eq. 27
                        rho = d2a_lim / d2af[i, j]

                    if (rho < 1.0 - 1.e-12):
                        # we may need to limit -- these quantities are at cell-centers
                        d3a_min = min(d3a[i, j - 1], d3a[i, j],
                                      d3a[i, j + 1], d3a[i, j + 2])
                        d3a_max = max(d3a[i, j - 1], d3a[i, j],
                                      d3a[i, j + 1], d3a[i, j + 2])

                        if (C3 * max(abs(d3a_min), abs(d3a_max)) <= (d3a_max - d3a_min)):
                            # limit
                            if (dafm[i, j] * dafp[i, j] < 0.0):
                                # Eqs. 29, 30
                                ar[i, j] = a[i, j] - rho * \
                                    dafm[i, j]  # note: typo in Eq 29
                                al[i, j + 1] = a[i, j] + rho * dafp[i, j]
                            elif (abs(dafm[i, j]) >= 2.0 * abs(dafp[i, j])):
                                # Eq. 31
                                ar[i, j] = a[i, j] - 2.0 * \
                                    (1.0 - rho) * dafp[i, j] - rho * dafm[i, j]
                            elif (abs(dafp[i, j]) >= 2.0 * abs(dafm[i, j])):
                                # Eq. 32
                                al[i, j + 1] = a[i, j] + 2.0 * \
                                    (1.0 - rho) * dafm[i, j] + rho * dafp[i, j]

                else:
                    # if Eqs. 24 or 25 didn't hold we still may need to limit
                    if (abs(dafm[i, j]) >= 2.0 * abs(dafp[i, j])):
                        ar[i, j] = a[i, j] - 2.0 * dafp[i, j]

                    if (abs(dafp[i, j]) >= 2.0 * abs(dafm[i, j])):
                        al[i, j + 1] = a[i, j] + 2.0 * dafm[i, j]

    return al, ar


@njit(cache=True)
def states_nolimit(a, qx, qy, ng, idir):
    r"""
    Predict the cell-centered state to the edges in one-dimension using the
    reconstructed slopes (and without slope limiting). We use a fourth-order
    Godunov method.

    Our convention here is that:

        ``al[i,j]``   will be :math:`al_{i-1/2,j}`,

        ``al[i+1,j]`` will be :math:`al_{i+1/2,j}`.

    Parameters
    ----------
    a : ndarray
        The cell-centered state.
    ng : int
        The number of ghost cells
    idir : int
        Are we predicting to the edges in the x-direction (1) or y-direction (2)?

    Returns
    -------
    out : ndarray, ndarray
        The state predicted to the left and right edges.
    """

    a_int = np.zeros((qx, qy))
    al = np.zeros((qx, qy))
    ar = np.zeros((qx, qy))

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    # we need interface values on all faces of the domain
    if (idir == 1):

        for i in range(ilo - 2, ihi + 3):
            for j in range(jlo - 1, jhi + 1):

                # interpolate to the edges
                a_int[i, j] = (7.0 / 12.0) * (a[i - 1, j] + a[i, j]) - \
                    (1.0 / 12.0) * (a[i - 2, j] + a[i + 1, j])

                al[i, j] = a_int[i, j]
                ar[i, j] = a_int[i, j]

    elif (idir == 2):

        for i in range(ilo - 1, ihi + 1):
            for j in range(jlo - 2, jhi + 3):

                # interpolate to the edges
                a_int[i, j] = (7.0 / 12.0) * (a[i, j - 1] + a[i, j]) - \
                    (1.0 / 12.0) * (a[i, j - 2] + a[i, j + 1])

                al[i, j] = a_int[i, j]
                ar[i, j] = a_int[i, j]

    return al, ar
