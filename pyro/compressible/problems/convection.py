"""A heat source in a layer at some height above the bottom will drive
convection in an adiabatically stratified atmosphere."""

import numpy as np

from pyro.util import msg

DEFAULT_INPUTS = "inputs.convection"

PROBLEM_PARAMS = {"convection.dens_base": 10.0,  # density at the base of the atmosphere
                  "convection.scale_height": 4.0,  # scale height of the isothermal atmosphere
                  "convection.y_height": 2.0,
                  "convection.thickness": 0.25,
                  "convection.e_rate": 0.1,
                  "convection.dens_cutoff": 0.01}


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

    scale_height = rp.get_param("convection.scale_height")
    dens_base = rp.get_param("convection.dens_base")
    dens_cutoff = rp.get_param("convection.dens_cutoff")

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
        elif dens[0, j] <= dens_cutoff + 1.e-30:
            p[:, j] = p[:, j-1]
        else:
            #p[:, j] = p[:, j-1] + 0.5*myg.dy*(dens[:, j] + dens[:, j-1])*grav
            p[:, j] = pres_base * (dens[:, j] / dens_base)**gamma

    # set the energy (P = cs2*dens) -- assuming zero velocity
    ener[:, :] = p[:, :]/(gamma - 1.0)

    # pairs of random numbers between [-1, 1]
    vel_pert = 2.0 * np.random.random_sample((myg.qx, myg.qy, 2)) - 1

    cs = np.sqrt(gamma * p / dens)

    # make vel_pert have M < 0.05
    vel_pert[:, :, 0] *= 0.05 * cs
    vel_pert[:, :, 1] *= 0.05 * cs

    idx = dens > 2 * dens_cutoff
    xmom[idx] = dens[idx] * vel_pert[idx, 0]
    ymom[idx] = dens[idx] * vel_pert[idx, 1]

    ener[:, :] += 0.5 * (xmom[:, :]**2 + ymom[:, :]**2) / dens[:, :]


def source_terms(myg, U, ivars, rp):
    """source terms to be added to the evolution"""

    S = myg.scratch_array(nvar=ivars.nvar)

    y_height = rp.get_param("convection.y_height")

    dist = np.abs(myg.y2d - y_height)

    e_rate = rp.get_param("convection.e_rate")
    thick = rp.get_param("convection.thickness")

    S[:, :, ivars.iener] = U[:, :, ivars.idens] * e_rate * np.exp(-(dist / thick)**2)
    return S


def finalize():
    """ print out any information to the user at the end of the run """
