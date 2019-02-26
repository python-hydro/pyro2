from __future__ import print_function

import sys
import mesh.patch as patch
import numpy as np


def init_data(my_data, rp):
    """ an init routine for unit testing """

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in sedov.py")
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
    dens[:, :] = 1.0
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0
    ener[:, :] = 2.5


def area(myg):
    return myg.dx * myg.dy + myg.scratch_array()


def h(idir, myg):
    if idir == 1:
        return myg.dy + myg.scratch_array()
    else:
        return myg.dx + myg.scratch_array()


def R(iface, myg, nvar, ixmom, iymom):
    R_fc = myg.scratch_array(nvar=(nvar, nvar))

    R_mat = np.eye(nvar)

    for i in range(myg.qx):
        for j in range(myg.qy):
            R_fc[i, j, :, :] = R_mat

    return R_fc


def finalize():
    """ print out any information to the user at the end of the run """
    pass
