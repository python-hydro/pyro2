"""A heat source at a point creates a plume that buoynantly rises"""

import numpy as np

from pyro.util import msg

DEFAULT_INPUTS = "inputs.plume"

PROBLEM_PARAMS = {"plume.dens_base": 10.0,  # density at the base of the atmosphere
                  "plume.scale_height": 4.0,  # scale height of the isothermal atmosphere
                  "plume.x_pert": 2.0,
                  "plume.y_pert": 2.0,
                  "plume.r_pert": 0.25,
                  "plume.e_rate": 0.1,
                  "plume.dens_cutoff": 0.01}


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

    scale_height = rp.get_param("plume.scale_height")
    dens_base = rp.get_param("plume.dens_base")
    dens_cutoff = rp.get_param("plume.dens_cutoff")

    # initialize the components, remember, that ener here is
    # rho*eint + 0.5*rho*v**2, where eint is the specific
    # internal energy (erg/g)
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0
    dens[:, :] = dens_cutoff

    # set the density to be stratified in the y-direction
    myg = my_data.grid

    p = myg.scratch_array()

    pres_base = scale_height*dens_base*abs(grav)

    for j in range(myg.jlo, myg.jhi+1):
        profile = 1.0 - (gamma-1.0)/gamma * myg.y[j]/scale_height
        if profile > 0.0:
            dens[:, j] = max(dens_base*(profile)**(1.0/(gamma-1.0)),
                             dens_cutoff)
        else:
            dens[:, j] = dens_cutoff

        if j == myg.jlo:
            p[:, j] = pres_base
        else:
            p[:, j] = p[:, j-1] + 0.5*myg.dy*(dens[:, j] + dens[:, j-1])*grav

    # set the energy (P = cs2*dens)
    ener[:, :] = p[:, :]/(gamma - 1.0) + \
                0.5*(xmom[:, :]**2 + ymom[:, :]**2)/dens[:, :]


def source_terms(myg, U, ivars, rp):
    """source terms to be added to the evolution"""

    S = myg.scratch_array(nvar=ivars.nvar)

    x_pert = rp.get_param("plume.x_pert")
    y_pert = rp.get_param("plume.y_pert")

    dist = np.sqrt((myg.x2d - x_pert)**2 +
                   (myg.y2d - y_pert)**2)

    e_rate = rp.get_param("plume.e_rate")
    r_pert = rp.get_param("plume.r_pert")

    S[:, :, ivars.iener] = U[:, :, ivars.idens] * e_rate * np.exp(-(dist / r_pert)**2)
    return S


def finalize():
    """ print out any information to the user at the end of the run """
