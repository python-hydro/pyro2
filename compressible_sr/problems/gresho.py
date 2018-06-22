from __future__ import print_function

import sys
import mesh.patch as patch
import numpy as np
from util import msg

import compressible_sr.eos as eos


def init_data(my_data, rp):
    """ initialize the Gresho vortex problem """

    msg.bold("initializing the Gresho vortex problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in bubble.py")
        print(my_data.__class__)
        sys.exit()

    # get the density and velocities
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    gamma = rp.get_param("eos.gamma")

    dens_base = rp.get_param("gresho.dens_base")

    R = rp.get_param("gresho.r")
    u0 = rp.get_param("gresho.u0")
    p0 = rp.get_param("gresho.p0")

    # initialize the components -- we'll get a psure too
    # but that is used only to initialize the base state
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0

    # set the density to be stratified in the y-direction
    myg = my_data.grid
    p = myg.scratch_array()

    dens[:, :] = dens_base

    p[:,:] = p0

    x_centre = 0.5 * (myg.x[0] + myg.x[-1])
    y_centre = 0.5 * (myg.y[0] + myg.y[-1])

    r = np.sqrt((myg.x2d - x_centre)**2 + (myg.y2d - y_centre)**2)

    p[r <= R] += 0.5 * (u0 * r[r<=R]/R)**2
    p[(r > R) & (r <= 2*R)] += u0**2 * (0.5 *(r[(r > R) & (r <= 2*R)]/R)**2 + 4 * (1 - r[(r > R) & (r <= 2*R)]/R + np.log(r[(r > R) & (r <= 2*R)]/R)))
    p[r > 2*R] += u0**2 * (4 * np.log(2) - 2)
    # print(p[r > 2*R])
    #
    uphi = np.zeros_like(p)
    uphi[r <= R] = u0 * r[r<=R]/R
    uphi[(r > R) & (r <= 2*R)] = u0 * (2 - r[(r > R) & (r <= 2*R)]/R)

    xmom[:,:] = -uphi[:,:] * (myg.y2d - y_centre) / r[:,:]
    ymom[:,:] = uphi[:,:] * (myg.x2d - x_centre) / r[:,:]

    # rhoe
    # ener[:, :] = p[:, :]/(gamma - 1.0) + \
    #             0.5*(xmom[:, :]**2 + ymom[:, :]**2)/dens[:, :]

    # dens[:,:] = p[:,:]/(eint[:,:]*(gamma - 1.0))

    rhoh = eos.rhoh_from_rho_p(gamma, dens, p)

    u = xmom
    v = ymom
    W = 1./np.sqrt(1. - u**2 - v**2)
    dens[:,:] *= W
    xmom[:, :] *= rhoh*W**2
    ymom[:, :] *= rhoh*W**2

    ener[:,:] = rhoh*W**2 - p - dens

    # print(ymom[5:-5, 5:-5])
    # exit()

def finalize():
    """ print out any information to the user at the end of the run """
    pass
