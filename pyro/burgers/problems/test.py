import sys

from pyro.mesh import patch
from pyro.util import msg


def init_data(myd, rp):
    """ initialize the burgers test problem """
    del rp  # this problem doesn't use runtime params

    msg.bold("initializing the burgers test problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(myd, patch.CellCenterData2d):
        print("ERROR: patch invalid in test.py")
        print(myd.__class__)
        sys.exit()

    u = myd.get_var("x-velocity")
    v = myd.get_var("y-velocity")

    u[:, :] = 3.0
    v[:, :] = 3.0

    # y = -x + 1

    index = myd.grid.y2d > -1.0 * myd.grid.x2d + 1.0

    u[index] = 1.0
    v[index] = 1.0


def finalize():
    """ print out any information to the user at the end of the run """
