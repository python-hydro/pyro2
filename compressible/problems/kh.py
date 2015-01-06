from __future__ import print_function

import math

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
    
    for i in range(myg.ilo, myg.ihi+1):
        for j in range(myg.jlo, myg.jhi+1):

            if myg.y[j] < yctr + 0.01*math.sin(10.0*math.pi*myg.x[i]/L_x):

                # lower half
                dens[i,j] = rho_1
                xmom[i,j] = rho_1*v_1
                ymom[i,j] = 0.0
                
            else:

                # upper half
                dens[i,j] = rho_2
                xmom[i,j] = rho_2*v_2
                ymom[i,j] = 0.0

            p = 1.0
            ener[i,j] = p/(gamma - 1.0) + \
                        0.5*(xmom[i,j]**2 + ymom[i,j]**2)/dens[i,j]


def finalize():
    """ print out any information to the user at the end of the run """
    pass
