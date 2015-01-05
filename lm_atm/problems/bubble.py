from __future__ import print_function

import sys
import mesh.patch as patch
import numpy
from util import msg

def init_data(my_data, base, rp):
    """ initialize the bubble problem """

    msg.bold("initializing the bubble problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in bubble.py")
        print(my_data.__class__)
        sys.exit()

    # get the density and velocities
    dens = my_data.get_var("density")
    xvel = my_data.get_var("x-velocity")
    yvel = my_data.get_var("y-velocity")
    eint = my_data.get_var("eint")

    grav = rp.get_param("lm-atmosphere.grav")

    gamma = rp.get_param("eos.gamma")

    scale_height = rp.get_param("bubble.scale_height")
    dens_base = rp.get_param("bubble.dens_base")
    dens_cutoff = rp.get_param("bubble.dens_cutoff")

    x_pert = rp.get_param("bubble.x_pert")
    y_pert = rp.get_param("bubble.y_pert")
    r_pert = rp.get_param("bubble.r_pert")
    pert_amplitude_factor = rp.get_param("bubble.pert_amplitude_factor")

    # initialize the components -- we'll get a pressure too
    # but that is used only to initialize the base state
    xvel[:,:] = 0.0
    yvel[:,:] = 0.0
    dens[:,:] = dens_cutoff

    # set the density to be stratified in the y-direction
    myg = my_data.grid
    pres = myg.scratch_array()

    j = myg.jlo
    for j in range(myg.jlo, myg.jhi+1):
        dens[:,j] = max(dens_base*numpy.exp(-myg.y[j]/scale_height),
                        dens_cutoff)

    cs2 = scale_height*abs(grav)

    # set the pressure (P = cs2*dens)
    pres = cs2*dens[:,:]
    
    for i in range(myg.ilo, myg.ihi+1):
        for j in range(myg.jlo, myg.jhi+1):

            r = numpy.sqrt((myg.x[i] - x_pert)**2  + (myg.y[j] - y_pert)**2)

            if r <= r_pert:
                # boost the specific internal energy, keeping the pressure
                # constant, by dropping the density
                eint[i,j] = pres[i,j]/(gamma - 1.0)/dens[i,j]
                eint[i,j] = eint[i,j]*pert_amplitude_factor

                dens[i,j] = pres[i,j]/(eint[i,j]*(gamma - 1.0))
    

    # do the base state
    base["rho0"] = numpy.mean(dens, axis=0)
    base["p0"] = numpy.mean(pres, axis=0)

    # redo the pressure via HSE
    for j in range(myg.jlo+1, myg.jhi):
        base["p0"][j] = base["p0"][j-1] + 0.5*myg.dy*(base["rho0"][j] + base["rho0"][j-1])*grav


def finalize():
    """ print out any information to the user at the end of the run """
    pass
