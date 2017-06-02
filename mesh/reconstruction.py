import mesh.patch as patch
import mesh.array_indexer as ai
import numpy as np

def limit(data, myg, idir, limiter):

    if limiter == 0:
        return nolimit(data, myg, idir)
    elif limiter == 1:
        return limit2(data, myg, idir)
    else:
        return limit4(data, myg, idir)


def nolimit(a, myg, idir):
    """ just a centered difference without any limiting """

    lda = myg.scratch_array()

    if idir == 1:
        lda.v(buf=2)[:,:] = 0.5*(a.ip(1, buf=2) - a.ip(-1, buf=2))
    elif idir == 2:
        lda.v(buf=2)[:,:] = 0.5*(a.jp(1, buf=2) - a.jp(-1, buf=2))

    return lda


def limit2(a, myg, idir):
    """ 2nd order monotonized central difference limiter """

    lda = myg.scratch_array()
    dc = myg.scratch_array()
    dl = myg.scratch_array()
    dr = myg.scratch_array()

    if idir == 1:
        dc.v(buf=2)[:,:] = 0.5*(a.ip(1, buf=2) - a.ip(-1, buf=2))
        dl.v(buf=2)[:,:] = a.ip(1, buf=2) - a.v(buf=2)
        dr.v(buf=2)[:,:] = a.v(buf=2) - a.ip(-1, buf=2)

    elif idir == 2:
        dc.v(buf=2)[:,:] = 0.5*(a.jp(1, buf=2) - a.jp(-1, buf=2))
        dl.v(buf=2)[:,:] = a.jp(1, buf=2) - a.v(buf=2)
        dr.v(buf=2)[:,:] = a.v(buf=2) - a.jp(-1, buf=2)

    d1 = 2.0*np.where(np.fabs(dl) < np.fabs(dr), dl, dr)
    dt = np.where(np.fabs(dc) < np.fabs(d1), dc, d1)
    lda.v(buf=myg.ng)[:,:] = np.where(dl*dr > 0.0, dt, 0.0)

    return lda
        

def limit4(a, myg, idir):
    """ 4th order monotonized central difference limiter """

    lda_tmp = limit2(a, myg, idir)

    lda = myg.scratch_array()
    dc = myg.scratch_array()
    dl = myg.scratch_array()
    dr = myg.scratch_array()

    if idir == 1:
        dc.v(buf=2)[:,:] = (2./3.)*(a.ip(1, buf=2) - a.ip(-1, buf=2) - 
                                    0.25*(lda_tmp.ip(1, buf=2) + lda_tmp.ip(-1, buf=2)))
        dl.v(buf=2)[:,:] = a.ip(1, buf=2) - a.v(buf=2)
        dr.v(buf=2)[:,:] = a.v(buf=2) - a.ip(-1, buf=2)

    elif idir == 2:
        dc.v(buf=2)[:,:] = (2./3.)*(a.jp(1, buf=2) - a.jp(-1, buf=2) - \
                                    0.25*(lda_tmp.jp(1, buf=2) + lda_tmp.jp(-1, buf=2)))
        dl.v(buf=2)[:,:] = a.jp(1, buf=2) - a.v(buf=2)
        dr.v(buf=2)[:,:] = a.v(buf=2) - a.jp(-1, buf=2)
    
    d1 = 2.0*np.where(np.fabs(dl) < np.fabs(dr), dl, dr)
    dt = np.where(np.fabs(dc) < np.fabs(d1), dc, d1)
    lda.v(buf=myg.ng)[:,:] = np.where(dl*dr > 0.0, dt, 0.0)

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
        t1.v(buf=2)[:,:] = abs(q.ip(1, n=ivars.ip, buf=2) - 
                               q.ip(-1, n=ivars.ip, buf=2))
        t2.v(buf=2)[:,:] = abs(q.ip(2, n=ivars.ip, buf=2) - 
                               q.ip(-2, n=ivars.ip, buf=2))

        z[:,:] = t1/np.maximum(t2, smallp)

        t2.v(buf=2)[:,:] = t1.v(buf=2)/np.minimum(q.ip(1, n=ivars.ip, buf=2), 
                                                  q.ip(-1, n=ivars.ip, buf=2))
        t1.v(buf=2)[:,:] = q.ip(-1, n=ivars.iu, buf=2) - q.ip(1, n=ivars.iu, buf=2)

    elif idir == 2:
        t1.v(buf=2)[:,:] = abs(q.jp(1, n=ivars.ip, buf=2) - 
                               q.jp(-1, n=ivars.ip, buf=2))
        t2.v(buf=2)[:,:] = abs(q.jp(2, n=ivars.ip, buf=2) - 
                               q.jp(-2, n=ivars.ip, buf=2))

        z[:,:] = t1/np.maximum(t2, smallp)

        t2.v(buf=2)[:,:] = t1.v(buf=2)/np.minimum(q.jp(1, n=ivars.ip, buf=2), 
                                                  q.jp(-1, n=ivars.ip, buf=2))
        t1.v(buf=2)[:,:] = q.jp(-1, n=ivars.iv, buf=2) - q.jp(1, n=ivars.iv, buf=2)

    xi.v(buf=myg.ng)[:,:] = np.minimum(1.0, np.maximum(0.0, 1.0 - (z - z0)/(z1 - z0)))

    xi[:,:] = np.where(np.logical_and(t1 > 0.0, t2 > delta), xi, 1.0)

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

    xi.v(buf=2)[:,:] = np.minimum(np.minimum(xi_x.v(buf=2), px),
                                  np.minimum(xi_y.v(buf=2), py))

    return xi



