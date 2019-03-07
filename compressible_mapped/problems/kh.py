from __future__ import print_function

import numpy as np
import sympy
from sympy.abc import x, y

import mesh.mapped as mapped
from util import msg


def init_data(my_data, rp):
    """ initialize the Kelvin-Helmholtz problem """

    msg.bold("initializing the Kelvin-Helmholtz problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, mapped.MappedCellCenterData2d):
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

    xmin_coord = rp.get_param("mesh.xmin")
    ymin_coord = rp.get_param("mesh.ymin")
    xmax_coord = rp.get_param("mesh.xmax")
    ymax_coord = rp.get_param("mesh.ymax")

    gamma = rp.get_param("eos.gamma")

    myg = my_data.grid

    xmin, ymin = myg.physical_coords(xmin_coord, ymin_coord)
    xmax, ymax = myg.physical_coords(xmax_coord, ymax_coord)

    dy = 0.025*(ymax - ymin)
    w0 = 0.01
    vm = 0.5 * (v_1 - v_2)
    rhom = 0.5 * (rho_1 - rho_2)

    X, Y = myg.physical_coords()

    idx1 = Y < 0.25*ymax
    idx2 = np.logical_and(Y >= 0.25*ymax, Y < 0.5*ymax)
    idx3 = np.logical_and(Y >= 0.5*ymax, Y < 0.75*ymax)
    idx4 = Y >= 0.75*ymax

    # we will initialize momemum as velocity for now

    # lower quarter
    dens[idx1] = rho_1 - rhom * np.exp((Y[idx1] - 0.25*ymax) / dy)
    xmom[idx1] = v_1 - vm * np.exp((Y[idx1] - 0.25*ymax) / dy)

    # second quarter
    dens[idx2] = rho_2 + rhom * np.exp((0.25*ymax - Y[idx2]) / dy)
    xmom[idx2] = v_2 + vm * np.exp((0.25*ymax - Y[idx2]) / dy)

    # third quarter
    dens[idx3] = rho_2 + rhom * np.exp((Y[idx3] - 0.75*ymax) / dy)
    xmom[idx3] = v_2 + vm * np.exp((Y[idx3] - 0.75*ymax) / dy)

    # fourth quarter
    dens[idx4] = rho_1 - rhom * np.exp((0.75*ymax - Y[idx4]) / dy)
    xmom[idx4] = v_1 - vm * np.exp((0.75*ymax - Y[idx4]) / dy)

    # upper half
    xmom[:, :] *= dens
    ymom[:, :] = dens * w0 * np.sin(4 * np.pi * X/(xmax-xmin))

    p = 2.5
    ener[:, :] = p / (gamma - 1.0) + 0.5 * (xmom[:, :]
                                            ** 2 + ymom[:, :]**2) / dens[:, :]


def sym_map(myg):

    return sympy.Matrix([-x + 4, 2*y])


def finalize():
    """ print out any information to the user at the end of the run """
    pass
