from __future__ import print_function

# import sys
# import mesh.fv as fv
import numpy as np
from util import msg
import sympy
from sympy.abc import x, y


def init_data(myd, rp):
    """initialize the acoustic_pulse problem.  This comes from
    McCourquodale & Coella 2011"""

    msg.bold("initializing the acoustic pulse problem...")

    # # make sure that we are passed a valid patch object
    # if not isinstance(myd, fv.FV2d):
    #     print("ERROR: patch invalid in acoustic_pulse.py")
    #     print(myd.__class__)
    #     sys.exit()

    # get the density, momenta, and energy as separate variables
    dens = myd.get_var("density")
    xmom = myd.get_var("x-momentum")
    ymom = myd.get_var("y-momentum")
    ener = myd.get_var("energy")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0

    gamma = rp.get_param("eos.gamma")

    rho0 = rp.get_param("acoustic_pulse.rho0")
    drho0 = rp.get_param("acoustic_pulse.drho0")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    myg = myd.grid

    xctr = 0.5 * (xmin + xmax) * myg.gamma_fcy
    yctr = 0.5 * (ymin + ymax) * myg.gamma_fcx

    dist = np.sqrt((myd.grid.x2d * myg.gamma_fcy - xctr)**2 +
                   (myd.grid.y2d * myg.gamma_fcx - yctr)**2)

    dens[:, :] = rho0
    idx = dist <= 0.5
    dens[idx] = rho0 + drho0 * \
        np.exp(-16 * dist[idx]**2) * np.cos(np.pi * dist[idx])**6

    p = (dens / rho0)**gamma
    ener[:, :] = p / (gamma - 1)


def area(myg):
    return 2 * myg.dx * myg.dy + myg.scratch_array()


def h(idir, myg):
    if idir == 1:
        return myg.dy + myg.scratch_array()
    else:
        return 2 * myg.dx + myg.scratch_array()


def R(iface, myg, nvar, ixmom, iymom):
    R_fc = myg.scratch_array(nvar=(nvar, nvar))

    R_mat = np.eye(nvar)

    for i in range(myg.qx):
        for j in range(myg.qy):
            R_fc[i, j, :, :] = R_mat

    return R_fc


def map(myg):

    xs_t = myg.x2d * 2
    ys_t = myg.y2d

    return xs_t, ys_t


def sym_map(myg):

    return sympy.Matrix([2 * x, y-1])


def finalize():
    """ print out any information to the user at the end of the run """
    pass
