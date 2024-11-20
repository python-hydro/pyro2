"""A test of the energy sources.  This uses a uniform domain and
slowly adds heat to the center over time."""

import numpy as np

from pyro.util import msg

DEFAULT_INPUTS = "inputs.heating"

PROBLEM_PARAMS = {"heating.rho_ambient": 1.0,  # ambient density
                  "heating.p_ambient": 10.0,  # ambient pressure
                  "heating.r_src": 0.1,  # physical size of the heating src
                  "heating.e_rate": 0.1}  # energy generation rate (energy / mass / time)


def init_data(my_data, rp):
    """ initialize the heating problem """

    if rp.get_param("driver.verbose"):
        msg.bold("initializing the heating problem...")

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    gamma = rp.get_param("eos.gamma")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    dens[:, :] = rp.get_param("heating.rho_ambient")
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0
    ener[:, :] = rp.get_param("heating.p_ambient") / (gamma - 1.0)


def source_terms(myg, U, ivars, rp):
    """source terms to be added to the evolution"""

    S = myg.scratch_array(nvar=ivars.nvar)

    xctr = 0.5 * (myg.xmin + myg.xmax)
    yctr = 0.5 * (myg.ymin + myg.ymax)

    dist = np.sqrt((myg.x2d - xctr)**2 +
                   (myg.y2d - yctr)**2)

    e_rate = rp.get_param("heating.e_rate")
    r_src = rp.get_param("heating.r_src")

    S[:, :, ivars.iener] = U[:, :, ivars.idens] * e_rate * np.exp(-(dist / r_src)**2)
    return S


def finalize():
    """ print out any information to the user at the end of the run """

    print("""
          The script analysis/sedov_compare.py can be used to analyze these
          results.  That will perform an average at constant radius and
          compare the radial profiles to the exact solution.  Sample exact
          data is provided as analysis/cylindrical-sedov.out
          """)
