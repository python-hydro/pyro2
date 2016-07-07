from __future__ import print_function

import sys
import mesh.patch as patch
import numpy as np
from util import msg
import math

def init_data(my_data, rp):
    """ initialize a smooth advection problem for testing convergence """

    msg.bold("initializing the advect problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
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
    dens.d[:,:] = 1.0
    xmom.d[:,:] = 0.0
    ymom.d[:,:] = 0.0


    gamma = rp.get_param("eos.gamma")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)

    # this is identical to the advection/smooth problem
    dens.d[:,:] = 1.0 + np.exp(-60.0*((my_data.grid.x2d-xctr)**2 + 
                                      (my_data.grid.y2d-yctr)**2))


    # velocity is diagonal
    u = 1.0
    v = 1.0
    xmom.d[:,:] = dens.d[:,:]*u
    ymom.d[:,:] = dens.d[:,:]*v

    # pressure is constant
    p = 1.0
    ener.d[:,:] = p/(gamma - 1.0) + 0.5*(xmom.d[:,:]**2 + ymom.d[:,:]**2)/dens.d[:,:]


def finalize():
    """ print out any information to the user at the end of the run """

    msg = """
          """

    print(msg)

