from __future__ import print_function

import sys
import numpy as np
import sympy
from sympy.abc import x, y

import mesh.mapped as mapped
from util import msg


def init_data(my_data, rp):
    """ initialize a smooth advection problem for testing convergence """

    msg.bold("initializing the advect problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, mapped.MappedCellCenterData2d):
        print("ERROR: patch invalid in advect.py")
        print(my_data.__class__)
        sys.exit()

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

    gamma = rp.get_param("eos.gamma")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    myg = my_data.grid

    X, Y = myg.physical_coords()

    xctr, yctr = 0.5 * \
        (myg.physical_coords(xmin, ymin) + myg.physical_coords(xmax, ymax))

    # this is identical to the advection/smooth problem
    dens[:, :] = 1.0 + np.exp(-60.0 * ((X - xctr)**2 +
                                       (Y - yctr)**2))

    # velocity is diagonal
    u = 1.0
    v = 1.0
    xmom[:, :] = dens[:, :] * u
    ymom[:, :] = dens[:, :] * v

    # pressure is constant
    p = 1.0
    ener[:, :] = p / (gamma - 1.0) + 0.5 * (xmom[:, :]
                                            ** 2 + ymom[:, :]**2) / dens[:, :]


def sym_map(myg):

    return sympy.Matrix([0.5*x + y, -x + 3])


def finalize():
    """ print out any information to the user at the end of the run """

    msg = """
          """

    print(msg)
