# this is the 4th-order McCorquodale and Colella method

import numpy as np

import advection_fv4.interface_f as interface_f
import compressible as comp
import compressible.interface_f as cf
import mesh.reconstruction as reconstruction
import mesh.array_indexer as ai

from util import msg

def flux_cons(ivars, idir, gamma, q):

    flux = q.g.scratch_array(nvar=ivars.nvar)

    if idir == 1:
        un = q[:,:,ivars.iu]
        ut = q[:,:,ivars.iv]
    else:
        ut = q[:,:,ivars.iu]
        un = q[:,:,ivars.iv]

    flux[:,:,ivars.idens] = q[:,:,ivars.irho]*un

    if idir == 1:
        flux[:,:,ivars.ixmom] = q[:,:,ivars.irho]*q[:,:,ivars.iu]**2 + q[:,:,ivars.ip]
        flux[:,:,ivars.iymom] = q[:,:,ivars.irho]*q[:,:,ivars.iv]*q[:,:,ivars.iu]
    else:
        flux[:,:,ivars.ixmom] = q[:,:,ivars.irho]*q[:,:,ivars.iu]*q[:,:,ivars.iv]
        flux[:,:,ivars.iymom] = q[:,:,ivars.irho]*q[:,:,ivars.iv]**2 + q[:,:,ivars.ip]

    flux[:,:,ivars.iener] = (q[:,:,ivars.ip]/(gamma - 1.0) + 0.5*q[:,:,ivars.irho]*(q[:,:,ivars.iu]**2 + q[:,:,ivars.iv]**2) + q[:,:,ivars.ip])*un

    if ivars.naux > 0:
        flux[:,:,ivars.irhox:ivars.irhox-1+ivars.naux] = q[:,:,ivars.irho]*q[:,:,ivars.ix:ivars.ix-1+ivars.naux]*un

    return flux

def fluxes(myd, rp, ivars, solid, tc):

    myg = myd.grid

    gamma = rp.get_param("eos.gamma")

    # get the cell-average data
    U_avg = myd.data

    # convert U from cell-centers to cell averages
    U_cc = np.zeros_like(U_avg)

    U_avg[:,:,ivars.idens] = myd.to_centers("density")
    U_avg[:,:,ivars.ixmom] = myd.to_centers("x-momentum")
    U_avg[:,:,ivars.iymom] = myd.to_centers("y-momentum")
    U_avg[:,:,ivars.iener] = myd.to_centers("energy")

    # compute the primitive variables of both the cell-center and averages
    q_avg = comp.cons_to_prim(U_avg, gamma, ivars, myd.grid)
    q_cc = comp.cons_to_prim(U_cc, gamma, ivars, myd.grid)

    # compute the 4th-order approximation to the cell-average primitive state
    q_fourth = myg.scratch_array(nvar=ivars.nq)
    for n in range(ivars.nq):
        q_fourth.v(n=n, buf=3)[:,:] = q_cc.v(n=n, buf=3) + myg.dx**2/24.0 * q_avg.lap(n=n, buf=3) 

    fluxes = []

    for idir in [1, 2]:

        # interpolate <W> to faces (with limiting)
        q_l = np.zeros_like(q_avg)
        q_r = np.zeros_like(q_avg)

        for n in range(ivars.nq):
            q_l[:,:,n], q_r[:,:,n] = interface_f.states(q_avg[:,:,n], myg.qx, myg.qy, myg.ng, idir)

        # solve the Riemann problem using the average face values
        if idir == 1:
            solid_lo = solid.xl
            solid_hi = solid.xr
        else:
            solid_lo = solid.yl
            solid_hi = solid.yr

        _q = cf.riemann_prim(idir, myg.qx, myg.qy, myg.ng, 
                             ivars.nvar, ivars.irho, ivars.iu, ivars.iv, ivars.ip, ivars.ix, ivars.naux,
                             solid_lo, solid_hi, 
                             gamma, q_l, q_r)

        q_int_avg = ai.ArrayIndexer(_q, grid=myg)

        # calculate the face-centered W
        q_int_fc = myg.scratch_array(nvar=ivars.nq)

        for n in range(ivars.nq):
            q_int_fc.v(n=n, buf=myg.ng-1)[:,:] = q_int_avg.v(n=n, buf=myg.ng-1) - myg.dx**2/24.0 * q_int_avg.lap(n=n, buf=myg.ng-1)

        # compute the final fluxes
        F_fc = flux_cons(ivars, idir, gamma, q_int_fc)
        F_avg = flux_cons(ivars, idir, gamma, q_int_avg)

        if idir == 1:
            F_x = myg.scratch_array(nvar=ivars.nvar)
            for n in range(ivars.nvar):
                F_x.v(n=n, buf=1)[:,:] = F_fc.v(n=n, buf=1) + 1.0/24.0 * (F_avg.jp(1, n=n, buf=1) - F_avg.v(n=n, buf=1) + F_avg.jp(-1, n=n, buf=1))
        else:
            F_y = myg.scratch_array(nvar=ivars.nvar)
            for n in range(ivars.nvar):
                F_y.v(n=n, buf=1)[:,:] = F_fc.v(n=n, buf=1) + 1.0/24.0 * (F_avg.ip(1, n=n, buf=1) - F_avg.v(n=n, buf=1) + F_avg.ip(-1, n=n, buf=1))

    return F_x, F_y
