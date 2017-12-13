# this is the 4th-order McCorquodale and Colella method

import compressible.interface_f as interface_f
import compressible as comp
import mesh.reconstruction as reconstruction
import mesh.array_indexer as ai

from util import msg

def fluxes(myd, rp, ivars, solid, tc):

    # get the cell-average data
    U_avg = myda.data

    # convert U from cell-centers to cell averages
    U_cc = np.zeros_like(U_avg)

    U_avg[:,:,ivars.idens] = my.data.to_centers("density")
    U_avg[:,:,ivars.ixmom] = my.data.to_centers("x-momentum")
    U_avg[:,:,ivars.iymom] = my.data.to_centers("y-momentum")
    U_avg[:,:,ivars.iener] = my.data.to_centers("energy")

    # compute the primitive variables of both the cell-center and averages
    q_avg = cons_to_prim(U_avg, gamma, ivars, myd.grid)
    q_cc = cons_to_prim(U_cc, gamma. ivars, myd.grid)

    # compute the 4th-order approximation to the cell-average primitive state
    q_fourth = q_cc + myg.dx**2/24.0 * q_avg.lap(buf=3)

    for idir in [1, 2]:
        # interpolate <W> to faces (with limiting)


        # solve the Riemann problem


        # calculate the face-centered W


        # compute the final fluxes


    return F_x, F_y
