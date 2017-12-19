"""Routines to help initialize cell-average values by oversampling the
initial conditions on a finer mesh and averaging down to the requested
mesh"""

import mesh.fv as fv

def get_finer(myd):

    mgf = myd.grid.fine_like(4)
    fd = fv.FV2d(mgf)

    for v in myd.names:
        fd.register_var(v, myd.BCs[v])

    fd.create()

    return fd

def average_down(myd, fd):
    """average the fine data from fd into the coarser object, myd"""

    for v in myd.names:
        var = myd.get_var(v)
        var[:,:] = fd.restrict(v, N=4)
