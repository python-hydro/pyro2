from __future__ import print_function

import sys
import mesh.patch as patch
import numpy as np
from util import msg
import matplotlib.pyplot as plt


def init_data(my_data, rp):
    """ initialize the sedov problem """

    msg.bold("initializing the logo problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in sedov.py")
        print(my_data.__class__)
        sys.exit()

    # create the logo
    myg = my_data.grid

    fig = plt.figure(2, (0.64, 0.64), dpi=100*myg.nx/64)
    fig.add_subplot(111)

    fig.text(0.5, 0.5, "pyro", transform=fig.transFigure, fontsize="16",
             horizontalalignment="center", verticalalignment="center")

    plt.axis("off")

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    logo = np.rot90(np.rot90(np.rot90((256-data[:, :, 1])/255.0)))

    # get the height, momenta as separate variables
    h = my_data.get_var("height")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    X = my_data.get_var("fuel")

    myg = my_data.grid

    # initialize the components
    h[:, :] = 1.0
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0

    # set the height in the logo zones to be really large
    logo_h = 2

    h.v()[:, :] = logo[:, :] * logo_h

    X.v()[:, :] = logo[:, :]

    corner_height = 2

    # explosion
    h[myg.ilo, myg.jlo] = corner_height
    h[myg.ilo, myg.jhi] = corner_height
    h[myg.ihi, myg.jlo] = corner_height
    h[myg.ihi, myg.jhi] = corner_height

    v = 1

    xmom[myg.ilo, myg.jlo] = v
    xmom[myg.ilo, myg.jhi] = v
    xmom[myg.ihi, myg.jlo] = -v
    xmom[myg.ihi, myg.jhi] = -v

    ymom[myg.ilo, myg.jlo] = v
    ymom[myg.ilo, myg.jhi] = -v
    ymom[myg.ihi, myg.jlo] = v
    ymom[myg.ihi, myg.jhi] = -v

    X[:, :] *= h


def finalize():
    """ print out any information to the user at the end of the run """
