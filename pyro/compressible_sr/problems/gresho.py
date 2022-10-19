import sys

import numpy as np

import pyro.compressible_sr.eos as eos
import pyro.mesh.patch as patch
from pyro.util import msg


def init_data(my_data, rp):
    """ initialize the Gresho vortex problem """

    msg.bold("initializing the Gresho vortex problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ErrrrOrr: patch invalid in bubble.py")
        print(my_data.__class__)
        sys.exit()

    # get the density and velocities
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    gamma = rp.get_param("eos.gamma")

    dens_base = rp.get_param("gresho.dens_base")

    rr = rp.get_param("gresho.r")
    u0 = rp.get_param("gresho.u0")
    p0 = rp.get_param("gresho.p0")

    # initialize the components -- we'll get a psure too
    # but that is used only to initialize the base state
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0

    # set the density to be stratified in the y-direction
    myg = my_data.grid
    pres = myg.scratch_array()

    dens[:, :] = dens_base

    pres[:, :] = p0

    x_centre = 0.5 * (myg.x[0] + myg.x[-1])
    y_centre = 0.5 * (myg.y[0] + myg.y[-1])

    rad = np.sqrt((myg.x2d - x_centre)**2 + (myg.y2d - y_centre)**2)

    pres[rad <= rr] += 0.5 * (u0 * rad[rad <= rr]/rr)**2
    pres[(rad > rr) & (rad <= 2*rr)] += u0**2 * \
        (0.5 * (rad[(rad > rr) & (rad <= 2*rr)]/rr)**2 +
        4 * (1 - rad[(rad > rr) & (rad <= 2*rr)]/rr +
        np.log(rad[(rad > rr) & (rad <= 2*rr)]/rr)))
    pres[rad > 2*rr] += u0**2 * (4 * np.log(2) - 2)
    # print(p[rad > 2*rr])
    #
    uphi = np.zeros_like(pres)
    uphi[rad <= rr] = u0 * rad[rad <= rr]/rr
    uphi[(rad > rr) & (rad <= 2*rr)] = u0 * (2 - rad[(rad > rr) & (rad <= 2*rr)]/rr)

    xmom[:, :] = -uphi[:, :] * (myg.y2d - y_centre) / rad[:, :]
    ymom[:, :] = uphi[:, :] * (myg.x2d - x_centre) / rad[:, :]

    # rhoe
    # enerad[:, :] = p[:, :]/(gamma - 1.0) + \
    #             0.5*(xmom[:, :]**2 + ymom[:, :]**2)/dens[:, :]

    # dens[:, :] = p[:, :]/(eint[:, :]*(gamma - 1.0))

    rhoh = eos.rhoh_from_rho_p(gamma, dens, pres)

    w_lor = 1./np.sqrt(1. - xmom**2 - ymom**2)
    dens[:, :] *= w_lor
    xmom[:, :] *= rhoh*w_lor**2
    ymom[:, :] *= rhoh*w_lor**2

    ener[:, :] = rhoh*w_lor**2 - pres - dens

    # print(ymom[5:-5, 5:-5])
    # exit()


def finalize():
    """ print out any information to the userad at the end of the run """
    pass
