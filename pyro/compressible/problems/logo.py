"""Generate the pyro logo!  The word "pyro" is written in the center
of the domain and perturbations are placed in the 4 corners to drive
converging shocks inward to scramble the logo.

"""

import matplotlib.pyplot as plt
import numpy as np

from pyro.util import msg

DEFAULT_INPUTS = "inputs.logo"

PROBLEM_PARAMS = {}


def init_data(my_data, rp):
    """ initialize the logo problem """

    if rp.get_param("driver.verbose"):
        msg.bold("initializing the logo problem...")

    # create the logo
    myg = my_data.grid

    fig = plt.figure(2, (0.64, 0.64), dpi=100*myg.nx/64)
    fig.add_subplot(111)

    fig.text(0.5, 0.5, "pyro", transform=fig.transFigure, fontsize="16",
             horizontalalignment="center", verticalalignment="center")

    plt.axis("off")

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    logo = np.rot90(np.rot90(np.rot90((256-data[:, :, 1])/255.0)))

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    myg = my_data.grid

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    dens[:, :] = 1.0
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0

    # set the density in the logo zones to be really large
    logo_dens = 50.0

    dens.v()[:, :] = logo[:, :] * logo_dens

    # pressure equilibrium
    gamma = rp.get_param("eos.gamma")

    p_ambient = 1.e-5
    ener[:, :] = p_ambient/(gamma - 1.0)

    # explosion
    ener[myg.ilo, myg.jlo] = 1.0
    ener[myg.ilo, myg.jhi] = 1.0
    ener[myg.ihi, myg.jlo] = 1.0
    ener[myg.ihi, myg.jhi] = 1.0


def finalize():
    """ print out any information to the user at the end of the run """

    print("""
          The script analysis/sedov_compare.py can be used to analyze these
          results.  That will perform an average at constant radius and
          compare the radial profiles to the exact solution.  Sample exact
          data is provided as analysis/cylindrical-sedov.out
          """)
