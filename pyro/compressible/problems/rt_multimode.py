"""A multi-mode Rayleigh-Taylor instability."""

import numpy as np

from pyro.util import msg

DEFAULT_INPUTS = "inputs.rt_multimode"

PROBLEM_PARAMS = {"rt_multimode.dens1": 1.0,
                  "rt_multimode.dens2": 2.0,
                  "rt_multimode.amp": 1.0,
                  "rt_multimode.sigma": 0.1,
                  "rt_multimode.nmodes": 10,
                  "rt_multimode.p0": 10.0}


def init_data(my_data, rp):
    """ initialize the rt problem """

    # see the random number generator
    rng = np.random.default_rng(12345)

    if rp.get_param("driver.verbose"):
        msg.bold("initializing the rt problem...")

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    gamma = rp.get_param("eos.gamma")

    grav = rp.get_param("compressible.grav")

    dens1 = rp.get_param("rt_multimode.dens1")
    dens2 = rp.get_param("rt_multimode.dens2")
    p0 = rp.get_param("rt_multimode.p0")
    amp = rp.get_param("rt_multimode.amp")
    sigma = rp.get_param("rt_multimode.sigma")
    nmodes = rp.get_param("rt_multimode.nmodes")

    # initialize the components, remember, that ener here is
    # rho*eint + 0.5*rho*v**2, where eint is the specific
    # internal energy (erg/g)
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0
    dens[:, :] = 0.0

    # set the density to be stratified in the y-direction
    myg = my_data.grid

    ycenter = 0.5*(myg.ymin + myg.ymax)

    p = myg.scratch_array()

    j = myg.jlo
    while j <= myg.jhi:
        if myg.y[j] < ycenter:
            dens[:, j] = dens1
            p[:, j] = p0 + dens1*grav*myg.y[j]

        else:
            dens[:, j] = dens2
            p[:, j] = p0 + dens1*grav*ycenter + dens2*grav*(myg.y[j] - ycenter)

        j += 1

    # add multiple modes to the vertical velocity
    L = myg.xmax - myg.xmin
    for k in range(1, nmodes+1):
        phase = rng.random() * 2 * np.pi
        mode_amp = amp * rng.random()
        ymom[:, :] += (mode_amp * np.cos(2.0 * np.pi * k*myg.x2d / L + phase) *
                       np.exp(-(myg.y2d - ycenter)**2 / sigma**2))
    ymom /= nmodes
    ymom *= dens

    # set the energy (P = cs2*dens)
    ener[:, :] = p[:, :]/(gamma - 1.0) + \
        0.5*(xmom[:, :]**2 + ymom[:, :]**2)/dens[:, :]


def finalize():
    """ print out any information to the user at the end of the run """
