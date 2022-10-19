# this is the 4th-order McCorquodale and Colella method

import numpy as np

import pyro.advection_fv4.interface as interface
import pyro.compressible as comp
import pyro.compressible.interface as cf
import pyro.mesh.array_indexer as ai
import pyro.mesh.reconstruction as reconstruction


def flux_cons(ivars, idir, gamma, q):

    flux = q.g.scratch_array(nvar=ivars.nvar)

    if idir == 1:
        un = q[:, :, ivars.iu]
    else:
        un = q[:, :, ivars.iv]

    flux[:, :, ivars.idens] = q[:, :, ivars.irho]*un

    if idir == 1:
        flux[:, :, ivars.ixmom] = q[:, :, ivars.irho]*q[:, :, ivars.iu]**2 + q[:, :, ivars.ip]
        flux[:, :, ivars.iymom] = q[:, :, ivars.irho]*q[:, :, ivars.iv]*q[:, :, ivars.iu]
    else:
        flux[:, :, ivars.ixmom] = q[:, :, ivars.irho]*q[:, :, ivars.iu]*q[:, :, ivars.iv]
        flux[:, :, ivars.iymom] = q[:, :, ivars.irho]*q[:, :, ivars.iv]**2 + q[:, :, ivars.ip]

    flux[:, :, ivars.iener] = (q[:, :, ivars.ip]/(gamma - 1.0) +
                               0.5*q[:, :, ivars.irho]*(q[:, :, ivars.iu]**2 +
                                                        q[:, :, ivars.iv]**2) + q[:, :, ivars.ip])*un

    if ivars.naux > 0:
        flux[:, :, ivars.irhox:ivars.irhox-1+ivars.naux] = \
            q[:, :, ivars.irho]*q[:, :, ivars.ix:ivars.ix-1+ivars.naux]*un

    return flux


def fluxes(myd, rp, ivars, solid, tc):

    alpha = 0.3
    beta = 0.3

    myg = myd.grid

    gamma = rp.get_param("eos.gamma")

    # get the cell-average data
    U_avg = myd.data

    # convert U from cell-centers to cell averages
    U_cc = np.zeros_like(U_avg)

    U_cc[:, :, ivars.idens] = myd.to_centers("density")
    U_cc[:, :, ivars.ixmom] = myd.to_centers("x-momentum")
    U_cc[:, :, ivars.iymom] = myd.to_centers("y-momentum")
    U_cc[:, :, ivars.iener] = myd.to_centers("energy")

    # compute the primitive variables of both the cell-center and averages
    q_bar = comp.cons_to_prim(U_avg, gamma, ivars, myd.grid)
    q_cc = comp.cons_to_prim(U_cc, gamma, ivars, myd.grid)

    # compute the 4th-order approximation to the cell-average primitive state
    q_avg = myg.scratch_array(nvar=ivars.nq)
    for n in range(ivars.nq):
        q_avg.v(n=n, buf=3)[:, :] = q_cc.v(n=n, buf=3) + myg.dx**2/24.0 * q_bar.lap(n=n, buf=3)

    # flattening -- there is a single flattening coefficient (xi) for all directions
    use_flattening = rp.get_param("compressible.use_flattening")

    if use_flattening:
        xi_x = reconstruction.flatten(myg, q_bar, 1, ivars, rp)
        xi_y = reconstruction.flatten(myg, q_bar, 2, ivars, rp)

        xi = reconstruction.flatten_multid(myg, q_bar, xi_x, xi_y, ivars)
    else:
        xi = 1.0

    # for debugging
    nolimit = 0

    for idir in [1, 2]:

        # interpolate <W> to faces (with limiting)
        q_l = myg.scratch_array(nvar=ivars.nq)
        q_r = myg.scratch_array(nvar=ivars.nq)

        if nolimit:
            for n in range(ivars.nq):

                if idir == 1:
                    qtmp = 7./12.*(q_avg.ip(-1, n=n, buf=1) + q_avg.v(n=n, buf=1)) - \
                           1./12.*(q_avg.ip(-2, n=n, buf=1) + q_avg.ip(1, n=n, buf=1))
                else:
                    qtmp = 7./12.*(q_avg.jp(-1, n=n, buf=1) + q_avg.v(n=n, buf=1)) - \
                           1./12.*(q_avg.jp(-2, n=n, buf=1) + q_avg.jp(1, n=n, buf=1))

                q_l.v(n=n, buf=1)[:, :] = qtmp
                q_r.v(n=n, buf=1)[:, :] = qtmp

        else:
            for n in range(ivars.nq):
                q_l[:, :, n], q_r[:, :, n] = interface.states(q_avg[:, :, n], myg.ng, idir)

            # apply flattening
            for n in range(ivars.nq):
                if idir == 1:
                    q_l.ip(1, n=n, buf=2)[:, :] = xi.v(buf=2)*q_l.ip(1, n=n, buf=2) + \
                        (1.0 - xi.v(buf=2))*q_avg.v(n=n, buf=2)
                    q_r.v(n=n, buf=2)[:, :] = xi.v(buf=2)*q_r.v(n=n, buf=2) + \
                        (1.0 - xi.v(buf=2))*q_avg.v(n=n, buf=2)
                else:
                    q_l.jp(1, n=n, buf=2)[:, :] = xi.v(buf=2)*q_l.jp(1, n=n, buf=2) + \
                        (1.0 - xi.v(buf=2))*q_avg.v(n=n, buf=2)
                    q_r.v(n=n, buf=2)[:, :] = xi.v(buf=2)*q_r.v(n=n, buf=2) + \
                        (1.0 - xi.v(buf=2))*q_avg.v(n=n, buf=2)

        _q = cf.riemann_prim(idir, myg.ng,
                             ivars.irho, ivars.iu, ivars.iv, ivars.ip, ivars.ix, ivars.naux,
                             0, 0,
                             gamma, q_l, q_r)

        q_int_avg = ai.ArrayIndexer(_q, grid=myg)

        # calculate the face-centered W using the transverse Laplacian
        q_int_fc = myg.scratch_array(nvar=ivars.nq)

        if idir == 1:
            for n in range(ivars.nq):
                q_int_fc.v(n=n, buf=myg.ng-1)[:, :] = q_int_avg.v(n=n, buf=myg.ng-1) - \
                    1.0/24.0 * (q_int_avg.jp(1, n=n, buf=myg.ng-1) -
                                2*q_int_avg.v(n=n, buf=myg.ng-1) +
                                q_int_avg.jp(-1, n=n, buf=myg.ng-1))
        else:
            for n in range(ivars.nq):
                q_int_fc.v(n=n, buf=myg.ng-1)[:, :] = q_int_avg.v(n=n, buf=myg.ng-1) - \
                    1.0/24.0 * (q_int_avg.ip(1, n=n, buf=myg.ng-1) -
                                2*q_int_avg.v(n=n, buf=myg.ng-1) +
                                q_int_avg.ip(-1, n=n, buf=myg.ng-1))

        # compute the final fluxes
        F_fc = flux_cons(ivars, idir, gamma, q_int_fc)
        F_avg = flux_cons(ivars, idir, gamma, q_int_avg)

        if idir == 1:
            F_x = myg.scratch_array(nvar=ivars.nvar)
            for n in range(ivars.nvar):
                F_x.v(n=n, buf=1)[:, :] = F_fc.v(n=n, buf=1) + \
                    1.0/24.0 * (F_avg.jp(1, n=n, buf=1) -
                                2*F_avg.v(n=n, buf=1) +
                                F_avg.jp(-1, n=n, buf=1))
        else:
            F_y = myg.scratch_array(nvar=ivars.nvar)
            for n in range(ivars.nvar):
                F_y.v(n=n, buf=1)[:, :] = F_fc.v(n=n, buf=1) + \
                    1.0/24.0 * (F_avg.ip(1, n=n, buf=1) -
                                2*F_avg.v(n=n, buf=1) +
                                F_avg.ip(-1, n=n, buf=1))

        # artificial viscosity McCorquodale & Colella Eq. 35, 36
        # first find face-centered div
        lam = myg.scratch_array()

        if idir == 1:
            lam.v(buf=1)[:, :] = (q_bar.v(buf=1, n=ivars.iu) -
                                  q_bar.ip(-1, buf=1, n=ivars.iu))/myg.dx + \
                                  0.25*(q_bar.jp(1, buf=1, n=ivars.iv) -
                                        q_bar.jp(-1, buf=1, n=ivars.iv) +
                                        q_bar.ip_jp(-1, 1, buf=1, n=ivars.iv) -
                                        q_bar.ip_jp(-1, -1, buf=1, n=ivars.iv))/myg.dy
        else:
            lam.v(buf=1)[:, :] = (q_bar.v(buf=1, n=ivars.iv) -
                                  q_bar.jp(-1, buf=1, n=ivars.iv))/myg.dy + \
                                  0.25*(q_bar.ip(1, buf=1, n=ivars.iu) -
                                        q_bar.ip(-1, buf=1, n=ivars.iu) +
                                        q_bar.ip_jp(1, -1, buf=1, n=ivars.iu) -
                                        q_bar.ip_jp(-1, -1, buf=1, n=ivars.iu))/myg.dx

        test = myg.scratch_array()
        test.v(buf=1)[:, :] = (myg.dx*lam.v(buf=1))**2 / \
                                (beta * gamma * q_bar.v(buf=1, n=ivars.ip) /
                                 q_bar.v(buf=1, n=ivars.irho))

        nu = myg.dx * lam * np.minimum(test, 1.0)
        nu[lam >= 0.0] = 0.0

        if idir == 1:
            for n in range(ivars.nvar):
                F_x.v(buf=1, n=n)[:, :] += alpha * nu.v(buf=1) * (U_avg.v(buf=1, n=n) - U_avg.ip(-1, buf=1, n=n))
        else:
            for n in range(ivars.nvar):
                F_y.v(buf=1, n=n)[:, :] += alpha * nu.v(buf=1) * (U_avg.v(buf=1, n=n) - U_avg.jp(-1, buf=1, n=n))

    return F_x, F_y
