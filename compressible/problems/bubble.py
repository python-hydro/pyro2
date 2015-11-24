from __future__ import print_function

import sys
import mesh.patch as patch
import numpy
from util import msg

def init_data(my_data, rp):
    """ initialize the bubble problem """

    msg.bold("initializing the bubble problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in bubble.py")
        print(my_data.__class__)
        sys.exit()

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    gamma = rp.get_param("eos.gamma")

    grav = rp.get_param("compressible.grav")

    scale_height = rp.get_param("bubble.scale_height")
    dens_base = rp.get_param("bubble.dens_base")
    dens_cutoff = rp.get_param("bubble.dens_cutoff")

    x_pert = rp.get_param("bubble.x_pert")
    y_pert = rp.get_param("bubble.y_pert")
    r_pert = rp.get_param("bubble.r_pert")
    pert_amplitude_factor = rp.get_param("bubble.pert_amplitude_factor")

    # initialize the components, remember, that ener here is
    # rho*eint + 0.5*rho*v**2, where eint is the specific
    # internal energy (erg/g)
    xmom.d[:,:] = 0.0
    ymom.d[:,:] = 0.0
    dens.d[:,:] = dens_cutoff

    # set the density to be stratified in the y-direction
    myg = my_data.grid

    for j in range(myg.jlo, myg.jhi+1):
        dens.d[:,j] = max(dens_base*numpy.exp(-myg.y[j]/scale_height),
                          dens_cutoff)

    cs2 = scale_height*abs(grav)

    # set the energy (P = cs2*dens)
    ener.d[:,:] = cs2*dens.d[:,:]/(gamma - 1.0) + \
                0.5*(xmom.d[:,:]**2 + ymom.d[:,:]**2)/dens.d[:,:]


    
    for i in range(myg.ilo, myg.ihi+1):
        for j in range(myg.jlo, myg.jhi+1):

            r = numpy.sqrt((myg.x[i] - x_pert)**2  + (myg.y[j] - y_pert)**2)

            if (r <= r_pert):
                # boost the specific internal energy, keeping the pressure
                # constant, by dropping the density
                eint = (ener.d[i,j] - 
                        0.5*(xmom.d[i,j]**2 - ymom.d[i,j]**2)/dens.d[i,j])/dens.d[i,j]

                pres = dens.d[i,j]*eint*(gamma - 1.0)

                eint = eint*pert_amplitude_factor
                dens.d[i,j] = pres/(eint*(gamma - 1.0))

                ener.d[i,j] = dens.d[i,j]*eint + \
                    0.5*(xmom.d[i,j]**2 + ymom.d[i,j]**2)/dens.d[i,j]

    

def finalize():
    """ print out any information to the user at the end of the run """
    pass
