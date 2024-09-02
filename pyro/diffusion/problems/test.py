import sys

from pyro.mesh import patch

DEFAULT_INPUTS = None


def init_data(my_data, rp):
    """ an init routine for unit testing """
    del rp  # this problem doesn't use runtime params

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in test.py")
        print(my_data.__class__)
        sys.exit()

    # get the density, momenta, and energy as separate variables
    phi = my_data.get_var("phi")
    phi[:, :] = 1.0


def finalize():
    """ print out any information to the user at the end of the run """
