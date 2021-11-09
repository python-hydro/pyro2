from __future__ import print_function

import sys
import mesh.patch as patch


def init_data(my_data, rp):
    """ an init routine for unit testing """

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in sedov.py")
        print(my_data.__class__)
        sys.exit()

    # get the hity, momenta, and energy as separate variables
    h = my_data.get_var("height")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")

    # initialize the components
    h[:, :] = 1.0
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0


def finalize():
    """ print out any information to the user at the end of the run """
    pass
