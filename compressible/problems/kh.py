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

    rho_1 = rp.get_param("kh.rho_1")
    v_1 = rp.get_param("kh.v_1")
    rho_2 = rp.get_param("kh.rho_2")
    v_2 = rp.get_param("kh.v_2")

    gamma = rp.get_param("eos.gamma")

    myg = my_data.grid

    dy = 0.025
    w0 = 0.01
    vm = 0.5*(v_1 - v_2)
    rhom = 0.5*(rho_1 - rho_2)

    idx1 = myg.y2d < 0.25
    idx2 = np.logical_and(myg.y2d >= 0.25, myg.y2d < 0.5)
    idx3 = np.logical_and(myg.y2d >= 0.5, myg.y2d < 0.75)
    idx4 = myg.y2d >= 0.75

    # we will initialize momemum as velocity for now

    # lower quarter
    dens[idx1] = rho_1 - rhom*np.exp((myg.y2d[idx1] - 0.25)/dy)
    xmom[idx1] = v_1 - vm*np.exp((myg.y2d[idx1] - 0.25)/dy)

    # second quarter
    dens[idx2] = rho_2 + rhom*np.exp((0.25 - myg.y2d[idx2])/dy)
    xmom[idx2] = v_2 + vm*np.exp((0.25 - myg.y2d[idx2])/dy)

    # third quarter
    dens[idx3] = rho_2 + rhom*np.exp((myg.y2d[idx3] - 0.75)/dy)
    xmom[idx3] = v_2 + vm*np.exp((myg.y2d[idx3] - 0.75)/dy)

    # fourth quarter
    dens[idx4] = rho_1 - rhom*np.exp((0.75 - myg.y2d[idx4])/dy)
    xmom[idx4] = v_1 - vm*np.exp((0.75 - myg.y2d[idx4])/dy)

    # upper half
    xmom[:, :] *= dens
    ymom[:, :] = dens * w0 * np.sin(4*np.pi*myg.x2d)

    p = 2.5
    ener[:, :] = p/(gamma - 1.0) + 0.5*(xmom[:, :]**2 + ymom[:, :]**2)/dens[:, :]


def finalize():
    """ print out any information to the user at the end of the run """
    pass
