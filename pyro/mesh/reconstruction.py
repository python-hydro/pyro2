"""Support for computing limited differences needed in reconstruction
of slopes in constructing interface states."""

import sys

import numpy as np


def limit(data, myg, idir, limiter):
    """ a single driver that calls the different limiters based on the value
    of the limiter input variable."""
    if limiter == 0:
        return nolimit(data, myg, idir)

    if limiter == 1:
        return limit2(data, myg, idir)

    return limit4(data, myg, idir)


def well_balance(q, myg, limiter, iv, grav):
    """subtract off the hydrostatic pressure before limiting.  Note, this
    only considers the y direction."""
    if limiter != 1:
        sys.exit("well-balanced only works for limiter == 1")

    p1 = myg.scratch_array()
    p1_jp1 = myg.scratch_array()
    p1_jm1 = myg.scratch_array()

    p1.v(buf=4)[:, :] = 0.0

    p1_jp1.v(buf=3)[:, :] = q.jp(1, buf=3, n=iv.ip) - (q.v(buf=3, n=iv.ip) +
            0.5*myg.dy*(q.v(buf=3, n=iv.irho) + q.jp(1, buf=3, n=iv.irho))*grav)

    p1_jm1.v(buf=3)[:, :] = q.jp(-1, buf=3, n=iv.ip) - (q.v(buf=3, n=iv.ip) -
            0.5*myg.dy*(q.v(buf=3, n=iv.irho) + q.jp(-1, buf=3, n=iv.irho))*grav)

    # now limit p1 using these -- this is the 2nd order MC limiter
    lda_tmp = myg.scratch_array()
    dc = myg.scratch_array()
    dl = myg.scratch_array()
    dr = myg.scratch_array()

    dc.v(buf=2)[:, :] = 0.5*(p1_jp1.v(buf=2) - p1_jm1.v(buf=2))
    dl.v(buf=2)[:, :] = p1_jp1.v(buf=2) - p1.v(buf=2)
    dr.v(buf=2)[:, :] = p1.v(buf=2) - p1_jm1.v(buf=2)

    d1 = 2.0*np.where(np.fabs(dl) < np.fabs(dr), dl, dr)
    dt = np.where(np.fabs(dc) < np.fabs(d1), dc, d1)
    lda_tmp.v(buf=myg.ng)[:, :] = np.where(dl*dr > 0.0, dt, 0.0)

    return lda_tmp


def nolimit(a, myg, idir):
    """ just a centered difference without any limiting """

    lda = myg.scratch_array()

    if idir == 1:
        lda.v(buf=2)[:, :] = 0.5*(a.ip(1, buf=2) - a.ip(-1, buf=2))
    elif idir == 2:
        lda.v(buf=2)[:, :] = 0.5*(a.jp(1, buf=2) - a.jp(-1, buf=2))

    return lda


def limit2(a, myg, idir):
    """ 2nd order monotonized central difference limiter """

    lda = myg.scratch_array()
    dc = myg.scratch_array()
    dl = myg.scratch_array()
    dr = myg.scratch_array()

    if idir == 1:
        dc.v(buf=2)[:, :] = 0.5*(a.ip(1, buf=2) - a.ip(-1, buf=2))
        dl.v(buf=2)[:, :] = a.ip(1, buf=2) - a.v(buf=2)
        dr.v(buf=2)[:, :] = a.v(buf=2) - a.ip(-1, buf=2)

    elif idir == 2:
        dc.v(buf=2)[:, :] = 0.5*(a.jp(1, buf=2) - a.jp(-1, buf=2))
        dl.v(buf=2)[:, :] = a.jp(1, buf=2) - a.v(buf=2)
        dr.v(buf=2)[:, :] = a.v(buf=2) - a.jp(-1, buf=2)

    d1 = 2.0*np.where(np.fabs(dl) < np.fabs(dr), dl, dr)
    dt = np.where(np.fabs(dc) < np.fabs(d1), dc, d1)
    lda.v(buf=myg.ng)[:, :] = np.where(dl*dr > 0.0, dt, 0.0)

    return lda


def limit4(a, myg, idir):
    """ 4th order monotonized central difference limiter """

    lda_tmp = limit2(a, myg, idir)

    lda = myg.scratch_array()
    dc = myg.scratch_array()
    dl = myg.scratch_array()
    dr = myg.scratch_array()

    if idir == 1:
        dc.v(buf=2)[:, :] = (2./3.)*(a.ip(1, buf=2) - a.ip(-1, buf=2) -
                                     0.25*(lda_tmp.ip(1, buf=2) + lda_tmp.ip(-1, buf=2)))
        dl.v(buf=2)[:, :] = a.ip(1, buf=2) - a.v(buf=2)
        dr.v(buf=2)[:, :] = a.v(buf=2) - a.ip(-1, buf=2)

    elif idir == 2:
        dc.v(buf=2)[:, :] = (2./3.)*(a.jp(1, buf=2) - a.jp(-1, buf=2) -
                                     0.25*(lda_tmp.jp(1, buf=2) + lda_tmp.jp(-1, buf=2)))
        dl.v(buf=2)[:, :] = a.jp(1, buf=2) - a.v(buf=2)
        dr.v(buf=2)[:, :] = a.v(buf=2) - a.jp(-1, buf=2)

    d1 = 2.0*np.where(np.fabs(dl) < np.fabs(dr), dl, dr)
    dt = np.where(np.fabs(dc) < np.fabs(d1), dc, d1)
    lda.v(buf=myg.ng)[:, :] = np.where(dl*dr > 0.0, dt, 0.0)

    return lda


def flatten(myg, q, idir, ivars, rp):
    """ compute the 1-d flattening coefficients """

    xi = myg.scratch_array()
    z = myg.scratch_array()
    t1 = myg.scratch_array()
    t2 = myg.scratch_array()

    delta = rp.get_param("compressible.delta")
    z0 = rp.get_param("compressible.z0")
    z1 = rp.get_param("compressible.z1")
    smallp = 1.e-10

    if idir == 1:
        t1.v(buf=2)[:, :] = abs(q.ip(1, n=ivars.ip, buf=2) -
                                q.ip(-1, n=ivars.ip, buf=2))
        t2.v(buf=2)[:, :] = abs(q.ip(2, n=ivars.ip, buf=2) -
                                q.ip(-2, n=ivars.ip, buf=2))

        z[:, :] = t1/np.maximum(t2, smallp)

        t2.v(buf=2)[:, :] = t1.v(buf=2)/np.minimum(q.ip(1, n=ivars.ip, buf=2),
                                                   q.ip(-1, n=ivars.ip, buf=2))
        t1.v(buf=2)[:, :] = q.ip(-1, n=ivars.iu, buf=2) - q.ip(1, n=ivars.iu, buf=2)

    elif idir == 2:
        t1.v(buf=2)[:, :] = abs(q.jp(1, n=ivars.ip, buf=2) -
                                q.jp(-1, n=ivars.ip, buf=2))
        t2.v(buf=2)[:, :] = abs(q.jp(2, n=ivars.ip, buf=2) -
                                q.jp(-2, n=ivars.ip, buf=2))

        z[:, :] = t1/np.maximum(t2, smallp)

        t2.v(buf=2)[:, :] = t1.v(buf=2)/np.minimum(q.jp(1, n=ivars.ip, buf=2),
                                                   q.jp(-1, n=ivars.ip, buf=2))
        t1.v(buf=2)[:, :] = q.jp(-1, n=ivars.iv, buf=2) - q.jp(1, n=ivars.iv, buf=2)

    xi.v(buf=myg.ng)[:, :] = np.minimum(1.0, np.maximum(0.0, 1.0 - (z - z0)/(z1 - z0)))

    xi[:, :] = np.where(np.logical_and(t1 > 0.0, t2 > delta), xi, 1.0)

    return xi


def flatten_multid(myg, q, xi_x, xi_y, ivars):
    """ compute the multidimensional flattening coefficient """

    xi = myg.scratch_array()

    px = np.where(q.ip(1, n=ivars.ip, buf=2) -
                  q.ip(-1, n=ivars.ip, buf=2) > 0,
                  xi_x.ip(-1, buf=2), xi_x.ip(1, buf=2))

    py = np.where(q.jp(1, n=ivars.ip, buf=2) -
                  q.jp(-1, n=ivars.ip, buf=2) > 0,
                  xi_y.jp(-1, buf=2), xi_y.jp(1, buf=2))

    xi.v(buf=2)[:, :] = np.minimum(np.minimum(xi_x.v(buf=2), px),
                                  np.minimum(xi_y.v(buf=2), py))

    return xi


# Constants for the WENO reconstruction
# NOTE: integer division laziness means this WILL fail on python2
C_3 = np.array([1, 2]) / 3

a_3 = np.array([[3, -1],
                [1,  1]]) / 2

sigma_3 = np.array([[[1, 0],
                     [-2, 1]],
                    [[1, 0],
                     [-2, 1]]])

C_5 = np.array([1, 6, 3]) / 10

a_5 = np.array([[11, -7, 2],
                [2, 5, -1],
                [-1, 5, 2]]) / 6

sigma_5 = np.array([[[40, 0, 0],
                     [-124, 100, 0],
                     [44, -76, 16]],
                    [[16, 0, 0],
                     [-52, 52, 0],
                     [20, -52, 16]],
                    [[16, 0, 0],
                     [-76, 100, 0],
                     [44, -124, 40]]]) / 12

C_all = {2: C_3,
         3: C_5}

a_all = {2: a_3,
         3: a_5}

sigma_all = {2: sigma_3,
             3: sigma_5}


def weno_upwind(q, order):
    """
    Perform upwinded (left biased) WENO reconstruction

    Parameters
    ----------

    q : np array
        input data
    order : int
        WENO order (k)

    Returns
    -------

    q_plus : np array
        data reconstructed to the right
    """
    a = a_all[order]
    C = C_all[order]
    sigma = sigma_all[order]
    epsilon = 1e-16
    alpha = np.zeros(order)
    beta = np.zeros(order)
    q_stencils = np.zeros(order)
    for k in range(order):
        for l in range(order):
            for m in range(l+1):
                beta[k] += sigma[k, l, m] * q[order-1+k-l] * q[order-1+k-m]
        alpha[k] = C[k] / (epsilon + beta[k]**2)
        for l in range(order):
            q_stencils[k] += a[k, l] * q[order-1+k-l]
    w = alpha / np.sum(alpha)

    return np.dot(w, q_stencils)


def weno(q, order):
    """
    Perform WENO reconstruction

    Parameters
    ----------

    q : np array
        input data with 3 ghost zones
    order : int
        WENO order (k)

    Returns
    -------

    q_plus, q_minus : np array
        data reconstructed to the right / left respectively
    """
    Npoints = q.shape
    q_minus = np.zeros_like(q)
    q_plus = np.zeros_like(q)

    for i in range(order, Npoints-order):
        q_plus[i] = weno_upwind(q[i+1-order:i+order], order)
        q_minus[i] = weno_upwind(q[i+order-1:i-order:-1], order)

    return q_minus, q_plus


def ppm_reconstruction(a, myg, idir, limiter):
    """
    The PPM reconstruction method proceeds as follows:

    1. First, we perform an estimation of the values
    of a on each interface x_{i+1/2}. In order to accomplish this
    particular goal, we fit a cuartic polynomial over the grid values:

    |----*----|----*----|----*----|----*----|
      a_{i-1}     a_i     a_{i+1}   a_{i+2}

    After the calculation of these values, defined as a_{i+1/2},
    we stored them in ai[i].

    2. Next, we define the values: a_{R,i-1/2} and a_{L,i+1/2},
    to the limits of a_{i-1/2} and a_{i+1/2} from the right and left
    of i-1/2 and i+1/2, respectively. Both values, a_{R,i+1/2} and
    a_{L,i+1/2} are fixed initially to a_{i+1/2}. However, after the
    limiting process, each interface will be double valued. Under this
    definition:

      |----------------------*----------------------|
    a_{R,i-1/2}                 a_i                   a_{L,i+1/2}

    with a_{R,i-1/2} stored in ar[i], and a_{L,i+1/2} in
    al[i].

    3. We apply the limit described in Colella and Woodward (1984) to
    reset the values of ar[i] and al[i].

    Parameters
    ----------
    a: np.array
        input cell data with 4 ghost cells.

    myg: <CellCentered2D class object>
        input mesh.

    idir: int
        desired reconstruction dimension.

    Returns
    -------

    al[i]: np.array
        the left limit interface values of the
        ppm-reconstruction

    ar[i]: np.array
        the right limit interface values of the
        ppm-reconstruction

    """

    al = myg.scratch_array()
    ar = myg.scratch_array()

    ai = myg.scratch_array()
    davl = myg.scratch_array()

    dl1 = myg.scratch_array()
    dl2 = myg.scratch_array()
    dc = myg.scratch_array()
    dr = myg.scratch_array()

    if idir == 1:

        davl = limit(a, myg, idir, limiter)

        ai.v(buf=2)[:, :] = a.v(buf=2) + 0.5*(a.ip(1, buf=2) - a.v(buf=2)) \
                               - (davl.ip(1, buf=2) - davl.v(buf=2))/6.0

        # From here we set ar and al to take the same value of ai. The values of ar and
        # al, will be reformated accordingly to the neigbours behaviour.

        al.v(buf=2)[:, :] = ai.v(buf=2)
        ar.v(buf=2)[:, :] = ai.ip(-1, buf=2)

        # From here, we can reset ar and al:

        dl1.v(buf=2)[:, :] = (al.v(buf=2) - a.v(buf=2)) * (a.v(buf=2) - ar.v(buf=2))
        dc.v(buf=2)[:, :] = a.v(buf=2)
        dr.v(buf=2)[:, :] = ar.v(buf=2)

        ar.v(buf=myg.ng)[:, :] = np.where(dl1 <= 0.0, dc, dr)  # First Condition

        dr.v(buf=2)[:, :] = al.v(buf=2)
        al.v(buf=myg.ng)[:, :] = np.where(dl1 <= 0.0, dc, dr)

        dl1.v(buf=2)[:, :] = (al.v(buf=2) - ar.v(buf=2)) \
                                 * (a.v(buf=2) - 0.5*(ar.v(buf=2) + al.v(buf=2)))

        dl2.v(buf=2)[:, :] = (al.v(buf=2) - ar.v(buf=2))**2 / 6.0
        dc.v(buf=2)[:, :] = 3.0*a.v(buf=2) - 2.0*al.v(buf=2)
        dr.v(buf=2)[:, :] = ar.v(buf=2)

        ar.v(buf=myg.ng)[:, :] = np.where(dl1 > dl2, dc, dr)  # Second Condition

        dc.v(buf=2)[:, :] = 3.0*a.v(buf=2) - 2.0*ar.v(buf=2)
        dr.v(buf=2)[:, :] = al.v(buf=2)

        al.v(buf=myg.ng)[:, :] = np.where(-dl2 > dl1, dc, dr)  # Third Condition

    elif idir == 2:

        davl = limit(a, myg, idir, limiter)

        ai.v(buf=2)[:, :] = a.v(buf=2) + 0.5*(a.jp(1, buf=2) - a.v(buf=2)) \
                               - (davl.jp(1, buf=2) - davl.v(buf=2))/6.0

        # From here we set ar and al to take the same value of ai. The values of ar and
        # al, will be reformated accordingly to the neigbours behaviour.
        al.v(buf=2)[:, :] = ai.v(buf=2)
        ar.v(buf=2)[:, :] = ai.jp(-1, buf=2)

        # From here, we can reset ar and al:
        dl1.v(buf=2)[:, :] = (al.v(buf=2) - a.v(buf=2)) * (a.v(buf=2) - ar.v(buf=2))
        dc.v(buf=2)[:, :] = a.v(buf=2)
        dr.v(buf=2)[:, :] = ar.v(buf=2)

        ar.v(buf=myg.ng)[:, :] = np.where(dl1 <= 0.0, dc, dr)  # First Condition

        dr.v(buf=2)[:, :] = al.v(buf=2)
        al.v(buf=myg.ng)[:, :] = np.where(dl1 <= 0.0, dc, dr)

        dl1.v(buf=2)[:, :] = (al.v(buf=2) - ar.v(buf=2)) \
                                 * (a.v(buf=2) - 0.5*(ar.v(buf=2) + al.v(buf=2)))

        dl2.v(buf=2)[:, :] = (al.v(buf=2) - ar.v(buf=2))**2 / 6.0
        dc.v(buf=2)[:, :] = 3.0*a.v(buf=2) - 2.0*al.v(buf=2)
        dr.v(buf=2)[:, :] = ar.v(buf=2)

        ar.v(buf=myg.ng)[:, :] = np.where(dl1 > dl2, dc, dr)  # Second Condition

        dc.v(buf=2)[:, :] = 3.0*a.v(buf=2) - 2.0*ar.v(buf=2)
        dr.v(buf=2)[:, :] = al.v(buf=2)

        al.v(buf=myg.ng)[:, :] = np.where(-dl2 > dl1, dc, dr)  # Third Condition

    return ar, al
