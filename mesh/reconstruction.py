import mesh.patch as patch
import mesh.reconstruction_f as reconstruction_f
import mesh.array_indexer as ai
import numpy as np

def limit(data, myg, idir, limiter):

    if limiter == 0:
        return nolimit(data, myg, idir)

    elif limiter < 10:
        if limiter == 1:
            return limit2(data, myg, idir)
        else:
            return limit4(data, myg, idir)
    else:
        ldax, lday = reconstruction_f.multid_limit(data, qx, qy, myg.ng)
        return ai.ArrayIndexer(d=ldax, grid=myg), ai.ArrayIndexer(d=lday, grid=myg)


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
