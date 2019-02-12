from __future__ import print_function

import sys
import mesh.patch as patch
import numpy as np
from util import msg


def init_data(cc_data, fcx_data, fcy_data, rp):
    """ initialize a smooth advection problem for testing convergence """

    msg.bold("initializing the advect problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(cc_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in advect.py")
        print(cc_data.__class__)
        sys.exit()

    cc_data.data[:,:,:] = 0.0

    # get the density, momenta, and energy as separate variables
    dens = cc_data.get_var("density")
    xmom = cc_data.get_var("x-momentum")
    ymom = cc_data.get_var("y-momentum")
    ener = cc_data.get_var("energy")
    bx = cc_data.get_var("x-magnetic-field")
    by = cc_data.get_var("y-magnetic-field")

    bx_fc = fcx_data.get_var("x-magnetic-field")
    by_fc = fcy_data.get_var("y-magnetic-field")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    dens[:, :] = 1.0
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0
    bx[:, :] = 0.00
    by[:, :] = 0.00

    gamma = rp.get_param("eos.gamma")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    xctr = 0.5 * (xmin + xmax)
    yctr = 0.5 * (ymin + ymax)

    # this is identical to the advection/smooth problem
    dens[:, :] = 1.0 + np.exp(-60.0 * ((cc_data.grid.x2d - xctr)**2 +
                                       (cc_data.grid.y2d - yctr)**2))

    # velocity is diagonal
    u = 1.0
    v = 1.0
    xmom[:, :] = dens[:, :] * u
    ymom[:, :] = dens[:, :] * v

    # pressure is constant
    p = 1.0
    ener[:, :] = p / (gamma - 1.0) + 0.5 * (u** 2 + v**2) * dens + \
        0.5 * (bx**2 + by**2)

    # make sure face centered-data is zero'd
    bx_fc[:,:] = 0
    by_fc[:,:] = 0


def finalize():
    """ print out any information to the user at the end of the run """

    msg = """
          """

    print(msg)
