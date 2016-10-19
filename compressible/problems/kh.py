from __future__ import print_function

import numpy as np

import mesh.patch as patch
from util import msg

def init_data(my_data, rp):
    """ initialize the Kelvin-Helmholtz problem """

    msg.bold("initializing the sedov problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print(my_data.__class__)
        msg.fail("ERROR: patch invalid in sedov.py")


    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    dens[:,:] = 1.0
    xmom[:,:] = 0.0
    ymom[:,:] = 0.0

    rho_1 = rp.get_param("kh.rho_1")
    v_1   = rp.get_param("kh.v_1")
    rho_2 = rp.get_param("kh.rho_2")
    v_2   = rp.get_param("kh.v_2")

    gamma = rp.get_param("eos.gamma")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    yctr = 0.5*(ymin + ymax)

    L_x = xmax - xmin

    myg = my_data.grid

    idx_l = myg.y2d < yctr + 0.01*np.sin(10.0*np.pi*myg.x2d/L_x)
    idx_h = myg.y2d >= yctr + 0.01*np.sin(10.0*np.pi*myg.x2d/L_x)

    # lower half
    dens[idx_l] = rho_1
    xmom[idx_l] = rho_1*v_1
    ymom[idx_l] = 0.0
                
    # upper half
    dens[idx_h] = rho_2
    xmom[idx_h] = rho_2*v_2
    ymom[idx_h] = 0.0

    p = 1.0
    ener[:,:] = p/(gamma - 1.0) + 0.5*(xmom[:,:]**2 + ymom[:,:]**2)/dens[:,:]


def finalize():
    """ print out any information to the user at the end of the run """
    pass
