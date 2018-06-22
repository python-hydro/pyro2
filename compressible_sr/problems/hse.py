from __future__ import print_function

import numpy as np

import sys
import mesh.patch as patch
from util import msg


def init_data(my_data, rp):
    """ initialize the HSE problem """

    msg.bold("initializing the HSE problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in hse.py")
        print(my_data.__class__)
        sys.exit()

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    gamma = rp.get_param("eos.gamma")

    grav = rp.get_param("compressible.grav")

    dens0 = rp.get_param("hse.dens0")
    print("dens0 = ", dens0)
    H = rp.get_param("hse.h")

    # isothermal sound speed (squared)
    cs2 = H*abs(grav)

    # initialize the components, remember, that ener here is
    # rho*eint + 0.5*rho*v**2, where eint is the specific
    # internal energy (erg/g)
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0
    dens[:, :] = 0.0

    # set the density to be stratified in the y-direction
    myg = my_data.grid

    p = myg.scratch_array()

    for j in range(myg.jlo, myg.jhi+1):
        dens[:, j] = dens0*np.exp(-myg.y[j]/H)
        if j == myg.jlo:
            p[:, j] = dens[:, j]*cs2
        else:
            p[:, j] = p[:, j-1] + 0.5*myg.dy*(dens[:, j] + dens[:, j-1])*grav

    # set the energy
    ener[:, :] = p[:, :]/(gamma - 1.0) + \
        0.5*(xmom[:, :]**2 + ymom[:, :]**2)/dens[:, :]


def finalize():
    """ print out any information to the user at the end of the run """
    pass
