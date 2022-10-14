import sys

import numpy

import pyro.mesh.patch as patch
from pyro.util import msg


def init_data(my_data, rp):
    """ initialize the smooth advection problem """

    msg.bold("initializing the smooth advection problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in smooth.py")
        print(my_data.__class__)
        sys.exit()

    dens = my_data.get_var("density")

    xmin = my_data.grid.xmin
    xmax = my_data.grid.xmax

    ymin = my_data.grid.ymin
    ymax = my_data.grid.ymax

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)

    dens[:, :] = 1.0 + numpy.exp(-60.0*((my_data.grid.x2d-xctr)**2 +
                                        (my_data.grid.y2d-yctr)**2))


def finalize():
    """ print out any information to the user at the end of the run """
    pass
