"""Initialize a buoyant bubble in a hydrostatic atmosphere.  This is
meant to be used to compare with the compressible solver bubble
problem.
"""

import numpy

from pyro.util import msg

DEFAULT_INPUTS = "inputs.bubble"

PROBLEM_PARAMS = {"bubble.dens_base": 10.0,  # density at the base of the atmosphere
                  "bubble.scale_height": 2.0,  # scale height of the isothermal atmosphere
                  "bubble.x_pert": 2.0,
                  "bubble.y_pert": 2.0,
                  "bubble.r_pert": 0.25,
                  "bubble.pert_amplitude_factor": 5.0,
                  "bubble.dens_cutoff": 0.01}


def init_data(my_data, base, rp):
    """ initialize the bubble problem """

    if rp.get_param("driver.verbose"):
        msg.bold("initializing the bubble problem...")

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
    xvel[:, :] = 0.0
    yvel[:, :] = 0.0
    dens[:, :] = dens_cutoff

    # set the density to be stratified in the y-direction
    myg = my_data.grid
    pres = myg.scratch_array()

    j = myg.jlo
    for j in range(myg.jlo, myg.jhi+1):
        dens[:, j] = max(dens_base*numpy.exp(-myg.y[j]/scale_height),
                         dens_cutoff)

    cs2 = scale_height*abs(grav)

    # set the pressure (P = cs2*dens)
    pres = cs2*dens
    eint[:, :] = pres/(gamma - 1.0)/dens

    # boost the specific internal energy, keeping the pressure
    # constant, by dropping the density
    r = numpy.sqrt((myg.x2d - x_pert)**2 + (myg.y2d - y_pert)**2)

    idx = r <= r_pert
    eint[idx] = eint[idx]*pert_amplitude_factor
    dens[idx] = pres[idx]/(eint[idx]*(gamma - 1.0))

    # do the base state
    base["rho0"].d[:] = numpy.mean(dens, axis=0)
    base["p0"].d[:] = numpy.mean(pres, axis=0)

    # redo the pressure via HSE
    for j in range(myg.jlo+1, myg.jhi):
        base["p0"].d[j] = base["p0"].d[j-1] + 0.5*myg.dy*(base["rho0"].d[j] + base["rho0"].d[j-1])*grav


def finalize():
    """ print out any information to the user at the end of the run """
