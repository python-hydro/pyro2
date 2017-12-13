# this is the 4th-order McCorquodale and Colella method

import numpy as np

import compressible.interface_f as interface_f
import compressible as comp
import mesh.reconstruction as reconstruction
import mesh.array_indexer as ai

from util import msg

def fluxes(myd, rp, ivars, solid, tc):

    # get the cell-average data
    U_avg = myd.data

    # convert U from cell-centers to cell averages
    U_cc = np.zeros_like(U_avg)

    U_avg[:,:,ivars.idens] = myd.to_centers("density")
    U_avg[:,:,ivars.ixmom] = myd.to_centers("x-momentum")
    U_avg[:,:,ivars.iymom] = myd.to_centers("y-momentum")
    U_avg[:,:,ivars.iener] = myd.to_centers("energy")

    # compute the primitive variables of both the cell-center and averages
    q_avg = cons_to_prim(U_avg, gamma, ivars, myd.grid)
    q_cc = cons_to_prim(U_cc, gamma. ivars, myd.grid)

    # compute the 4th-order approximation to the cell-average primitive state
    q_fourth = q_cc + myg.dx**2/24.0 * q_avg.lap(buf=3)

    fluxes = []

    for idir in [1, 2]:

        # interpolate <W> to faces (with limiting)
        q_l = np.zeros_like(q_avg)
        q_r = np.zeros_like(q_avg)

        for n in q_avg.shape[-1]:
            q_l[:,:,n], q_r[:,:,n] = interface_f.states(q_avg[:,:,n], myg.qx, myg.qy, myg.ng, idir)

        # solve the Riemann problem using the average face values
        q_int_avg = riemann_prim(idir, myg.qx. myg.qy, myg.ng, 
                                 ivars.nvar, ivars.irho, ivars.iu, ivars.iv, ivars.ip,
                                 gamma, q_l, q_r)

        # calculate the face-centered W
        for n in q_int_avg.shape[-1]:
            q_int_fc[:,:,n] = q_int_avg[:,:,n] - myg.dx**2/24.0 * q_int_avg.lap(buf=myg.ng-1)

        # compute the final fluxes
        F_fc = F_cons(q_int_fc)
        F_avg = F_cons(q_int_avg)

        F_x.v()[:,:] = F_fc.v() + 1.0/24.0 * (F_avg.ip(1) - F_avg.v() + F_avg.ip(-1))

    return F_x, F_y
