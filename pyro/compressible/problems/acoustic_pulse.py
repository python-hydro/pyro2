"""The acoustic pulse problem described in McCorquodale & Colella
2011.  This uses a uniform background and a small pressure
perturbation that drives a low Mach number soundwave.  This problem is
useful for testing convergence of a compressible solver.

"""

import numpy as np

from pyro.util import msg

DEFAULT_INPUTS = "inputs.acoustic_pulse"

PROBLEM_PARAMS = {"acoustic_pulse.rho0": 1.4,
                  "acoustic_pulse.drho0": 0.14}


def init_data(myd, rp):
    """initialize the acoustic_pulse problem.  This comes from
    McCorquodale & Coella 2011"""

    if rp.get_param("driver.verbose"):
        msg.bold("initializing the acoustic pulse problem...")

    # get the density, momenta, and energy as separate variables
    dens = myd.get_var("density")
    xmom = myd.get_var("x-momentum")
    ymom = myd.get_var("y-momentum")
    ener = myd.get_var("energy")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0

    gamma = rp.get_param("eos.gamma")

    rho0 = rp.get_param("acoustic_pulse.rho0")
    drho0 = rp.get_param("acoustic_pulse.drho0")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)

    dist = np.sqrt((myd.grid.x2d - xctr)**2 +
                   (myd.grid.y2d - yctr)**2)

    dens[:, :] = rho0
    idx = dist <= 0.5
    dens[idx] = rho0 + drho0*np.exp(-16*dist[idx]**2) * np.cos(np.pi*dist[idx])**6

    p = (dens/rho0)**gamma
    ener[:, :] = p/(gamma - 1)


def finalize():
    """ print out any information to the user at the end of the run """
