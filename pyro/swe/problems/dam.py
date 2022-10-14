import sys

import pyro.mesh.patch as patch
from pyro.util import msg


def init_data(my_data, rp):
    """ initialize the dam problem """

    msg.bold("initializing the dam problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in dam.py")
        print(my_data.__class__)
        sys.exit()

    # get the dam parameters
    h_left = rp.get_param("dam.h_left")
    h_right = rp.get_param("dam.h_right")

    u_left = rp.get_param("dam.u_left")
    u_right = rp.get_param("dam.u_right")

    # get the height, momenta, and energy as separate variables
    h = my_data.get_var("height")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    X = my_data.get_var("fuel")

    # initialize the components
    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    direction = rp.get_param("dam.direction")

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)

    myg = my_data.grid

    if direction == "x":

        # left
        idxl = myg.x2d <= xctr

        h[idxl] = h_left
        xmom[idxl] = h_left*u_left
        ymom[idxl] = 0.0
        X[idxl] = 1.0

        # right
        idxr = myg.x2d > xctr

        h[idxr] = h_right
        xmom[idxr] = h_right*u_right
        ymom[idxr] = 0.0
        X[idxr] = 0.0

    else:

        # bottom
        idxb = myg.y2d <= yctr

        h[idxb] = h_left
        xmom[idxb] = 0.0
        ymom[idxb] = h_left*u_left
        X[idxb] = 1.0

        # top
        idxt = myg.y2d > yctr

        h[idxt] = h_right
        xmom[idxt] = 0.0
        ymom[idxt] = h_right*u_right
        X[idxt] = 0.0

    X[:, :] *= h


def finalize():
    """ print out any information to the user at the end of the run """

    msg = """
          The script analysis/dam_compare.py can be used to compare
          this output to the exact solution.
          """

    print(msg)
