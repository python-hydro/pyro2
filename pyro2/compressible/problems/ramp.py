from __future__ import print_function

import sys

import numpy as np
import math

import mesh.patch as patch
from util import msg


def init_data(my_data, rp):
    """ initialize the double Mach reflection problem """

    msg.bold("initializing the double Mach reflection problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in ramp.py")
        print(my_data.__class__)
        sys.exit()

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    # initialize the components, remember, that ener here is
    # rho*eint + 0.5*rho*v**2, where eint is the specific
    # internal energy (erg/g)

    r_l = rp.get_param("ramp.rhol")
    u_l = rp.get_param("ramp.ul")
    v_l = rp.get_param("ramp.vl")
    p_l = rp.get_param("ramp.pl")

    r_r = rp.get_param("ramp.rhor")
    u_r = rp.get_param("ramp.ur")
    v_r = rp.get_param("ramp.vr")
    p_r = rp.get_param("ramp.pr")

    gamma = rp.get_param("eos.gamma")

    energy_l = p_l/(gamma - 1.0) + 0.5*r_l*(u_l*u_l + v_l*v_l)
    energy_r = p_r/(gamma - 1.0) + 0.5*r_r*(u_r*u_r + v_r*v_r)

    # there is probably an easier way to do this, but for now, we
    # will just do an explicit loop.  Also, we really want to set
    # the pressue and get the internal energy from that, and then
    # compute the total energy (which is what we store).  For now
    # we will just fake this

    myg = my_data.grid
    dens[:, :] = 1.4

    for j in range(myg.jlo, myg.jhi+1):
        cy_up = myg.y[j] + 0.5*myg.dy*math.sqrt(3)
        cy_down = myg.y[j] - 0.5*myg.dy*math.sqrt(3)
        cy = np.array([cy_down, cy_up])

        for i in range(myg.ilo, myg.ihi+1):
            dens[i, j] = 0.0
            xmom[i, j] = 0.0
            ymom[i, j] = 0.0
            ener[i, j] = 0.0

            sf_up = math.tan(math.pi/3.0)*(myg.x[i] + 0.5*myg.dx*math.sqrt(3)-1.0/6.0)
            sf_down = math.tan(math.pi/3.0)*(myg.x[i] - 0.5*myg.dx*math.sqrt(3)-1.0/6.0)
            sf = np.array([sf_down, sf_up])   # initial shock front

            for y in cy:
                for shockfront in sf:
                    if y >= shockfront:
                        dens[i, j] = dens[i, j] + 0.25*r_l
                        xmom[i, j] = xmom[i, j] + 0.25*r_l*u_l
                        ymom[i, j] = ymom[i, j] + 0.25*r_l*v_l
                        ener[i, j] = ener[i, j] + 0.25*energy_l
                    else:
                        dens[i, j] = dens[i, j] + 0.25*r_r
                        xmom[i, j] = xmom[i, j] + 0.25*r_r*u_r
                        ymom[i, j] = ymom[i, j] + 0.25*r_r*v_r
                        ener[i, j] = ener[i, j] + 0.25*energy_r


def finalize():
    """ print out any information to the user at the end of the run """
    pass
