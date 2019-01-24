from __future__ import print_function

import sys
import mesh.patch as patch
import numpy as np
from util import msg


def init_data(my_data, rp):
    """ initialize a weak magnetic field loop """

    msg.bold("initializing the loop problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in loop.py")
        print(my_data.__class__)
        sys.exit()

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")
    bx = my_data.get_var("x-magnetic-field")
    by = my_data.get_var("y-magnetic-field")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    dens[:, :] = 1.0
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0
    bx[:, :] = 0.0
    by[:, :] = 0.0

    gamma = rp.get_param("eos.gamma")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    v0 = rp.get_param("loop.v0")
    R = rp.get_param("loop.r")
    A0 = rp.get_param("loop.a0")

    # velocity
    u = v0 * 2 / np.sqrt(5)
    v = v0 / np.sqrt(5)
    xmom[:, :] = dens[:, :] * u
    ymom[:, :] = dens[:, :] * v

    # magnetic field
    myg = my_data.grid
    r = np.sqrt(myg.x2d**2 + myg.y2d**2)
    # Az = np.zeros_like(bx)
    # Az[r <= R] = A0 * (R - r)

    # calculate Bx, By from magnetic vector potential Az
    bx[r <= R] = -A0 * myg.y2d[r <= R] / r[r <= R]
    by[r <= R] = A0 * myg.x2d[r <= R] / r[r <= R]

    # pressure is constant
    p = 1.0
    ener[:, :] = p / (gamma - 1.0) + \
        0.5 * (xmom ** 2 + ymom**2) / dens + \
        0.5 * (bx**2 + by**2)


def finalize():
    """ print out any information to the user at the end of the run """

    msg = """
          """

    print(msg)
