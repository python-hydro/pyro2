import sys

import pyro.mesh.patch as patch
from pyro.util import msg


def init_data(myd, rp):
    """ initialize the tophat advection problem """

    msg.bold("initializing the tophat advection problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(myd, patch.CellCenterData2d):
        print("ERROR: patch invalid in tophat.py")
        print(myd.__class__)
        sys.exit()

    dens = myd.get_var("density")

    xmin = myd.grid.xmin
    xmax = myd.grid.xmax

    ymin = myd.grid.ymin
    ymax = myd.grid.ymax

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)

    dens[:, :] = 0.0

    R = 0.1

    inside = (myd.grid.x2d - xctr)**2 + (myd.grid.y2d - yctr)**2 < R**2

    dens[inside] = 1.0


def finalize():
    """ print out any information to the user at the end of the run """
    pass
