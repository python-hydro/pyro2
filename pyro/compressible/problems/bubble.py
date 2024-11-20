"""A buoyant perturbation (bubble) is placed in an isothermal
hydrostatic atmosphere (plane-parallel).  It will rise and deform (due
to shear)"""

import numpy as np

from pyro.util import msg

DEFAULT_INPUTS = "inputs.bubble"

PROBLEM_PARAMS = {"bubble.dens_base": 10.0,  # density at the base of the atmosphere
                  "bubble.scale_height": 2.0,  # scale height of the isothermal atmosphere
                  "bubble.x_pert": 2.0,
                  "bubble.y_pert": 2.0,
                  "bubble.r_pert": 0.25,
                  "bubble.pert_amplitude_factor": 5.0,
                  "bubble.dens_cutoff": 0.01}


def init_data(my_data, rp):
    """ initialize the bubble problem """

    if rp.get_param("driver.verbose"):
        msg.bold("initializing the bubble problem...")

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
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0
    dens[:, :] = dens_cutoff

    # set the density to be stratified in the y-direction
    myg = my_data.grid

    p = myg.scratch_array()

    cs2 = scale_height*abs(grav)

    for j in range(myg.jlo, myg.jhi+1):
        dens[:, j] = max(dens_base*np.exp(-myg.y[j]/scale_height),
                        dens_cutoff)
        if j == myg.jlo:
            p[:, j] = dens[:, j]*cs2
        else:
            p[:, j] = p[:, j-1] + 0.5*myg.dy*(dens[:, j] + dens[:, j-1])*grav

    # set the energy (P = cs2*dens)
    ener[:, :] = p[:, :]/(gamma - 1.0) + \
                0.5*(xmom[:, :]**2 + ymom[:, :]**2)/dens[:, :]

    r = np.sqrt((myg.x2d - x_pert)**2 + (myg.y2d - y_pert)**2)
    idx = r <= r_pert

    # boost the specific internal energy, keeping the pressure
    # constant, by dropping the density
    eint = (ener[idx] - 0.5*(xmom[idx]**2 - ymom[idx]**2)/dens[idx])/dens[idx]

    pres = dens[idx]*eint*(gamma - 1.0)

    eint = eint*pert_amplitude_factor
    dens[idx] = pres/(eint*(gamma - 1.0))

    ener[idx] = dens[idx]*eint + 0.5*(xmom[idx]**2 + ymom[idx]**2)/dens[idx]


def finalize():
    """ print out any information to the user at the end of the run """
