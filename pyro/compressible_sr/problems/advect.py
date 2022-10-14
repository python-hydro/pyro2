import sys

import numpy as np

import pyro.compressible_sr.eos as eos
import pyro.mesh.patch as patch
from pyro.util import msg


def init_data(my_data, rp):
    """ initialize a smooth advection problem for testing convergence """

    msg.bold("initializing the advect problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in advect.py")
        print(my_data.__class__)
        sys.exit()

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    dens[:, :] = 0.2
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0

    gamma = rp.get_param("eos.gamma")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)

    # this is identical to the advection/smooth problem
    dens[:, :] = 0.2 * (1 +
                 np.exp(-60.0 * ((my_data.grid.x2d-xctr)**2 +
                                 (my_data.grid.y2d-yctr)**2)))

    # velocity is diagonal
    u = 0.4
    v = 0.4

    # pressure is constant
    p = 0.2

    rhoh = eos.rhoh_from_rho_p(gamma, dens, p)

    W = 1./np.sqrt(1-u**2-v**2)
    dens[:, :] *= W
    xmom[:, :] = rhoh[:, :]*u*W**2
    ymom[:, :] = rhoh[:, :]*v*W**2

    ener[:, :] = rhoh[:, :]*W**2 - p - dens[:, :]


def finalize():
    """ print out any information to the user at the end of the run """

    msg = """
          """

    print(msg)
