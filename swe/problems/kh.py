from __future__ import print_function

import numpy as np

import mesh.patch as patch
from util import msg


def init_data(my_data, rp):
    """ initialize the Kelvin-Helmholtz problem """

    msg.bold("initializing the Kelvin-Helmholtz problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print(my_data.__class__)
        msg.fail("ERROR: patch invalid in kh.py")

    # get the heightity, momenta, and energy as separate variables
    height = my_data.get_var("height")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    X = my_data.get_var("fuel")

    # initialize the components, remember, that ener here is h*eint
    # + 0.5*h*v**2, where eint is the specific internal energy
    # (erg/g)
    height[:, :] = 1.0
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0

    h_1 = rp.get_param("kh.h_1")
    v_1 = rp.get_param("kh.v_1")
    h_2 = rp.get_param("kh.h_2")
    v_2 = rp.get_param("kh.v_2")

    myg = my_data.grid

    dy = 0.025
    w0 = 0.01
    vm = 0.5*(v_1 - v_2)
    hm = 0.5*(h_1 - h_2)

    idx1 = myg.y2d < 0.25
    idx2 = np.logical_and(myg.y2d >= 0.25, myg.y2d < 0.5)
    idx3 = np.logical_and(myg.y2d >= 0.5, myg.y2d < 0.75)
    idx4 = myg.y2d >= 0.75

    # we will initialize momemum as velocity for now

    # lower quarter
    height[idx1] = h_1 - hm*np.exp((myg.y2d[idx1] - 0.25)/dy)
    xmom[idx1] = v_1 - vm*np.exp((myg.y2d[idx1] - 0.25)/dy)
    X[idx1] = 1 - 0.5*np.exp((myg.y2d[idx1] - 0.25)/dy)

    # second quarter
    height[idx2] = h_2 + hm*np.exp((0.25 - myg.y2d[idx2])/dy)
    xmom[idx2] = v_2 + vm*np.exp((0.25 - myg.y2d[idx2])/dy)
    X[idx2] = 0.5*np.exp((0.25 - myg.y2d[idx2])/dy)

    # third quarter
    height[idx3] = h_2 + hm*np.exp((myg.y2d[idx3] - 0.75)/dy)
    xmom[idx3] = v_2 + vm*np.exp((myg.y2d[idx3] - 0.75)/dy)
    X[idx3] = 0.5*np.exp((myg.y2d[idx3] - 0.75)/dy)

    # fourth quarter
    height[idx4] = h_1 - hm*np.exp((0.75 - myg.y2d[idx4])/dy)
    xmom[idx4] = v_1 - vm*np.exp((0.75 - myg.y2d[idx4])/dy)
    X[idx4] = 1 - 0.5*np.exp((0.75 - myg.y2d[idx4])/dy)

    # upper half
    xmom[:, :] *= height
    ymom[:, :] = height * w0 * np.sin(4*np.pi*myg.x2d)
    X[:, :] *= height


def finalize():
    """ print out any information to the user at the end of the run """
    pass
