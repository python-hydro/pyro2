import sys

import matplotlib.pyplot as plt
import numpy as np

import pyro.compressible_sr.eos as eos
import pyro.mesh.patch as patch
from pyro.util import msg


def init_data(my_data, rp):
    """ initialize the sedov problem """

    msg.bold("initializing the logo problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in sedov.py")
        print(my_data.__class__)
        sys.exit()

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
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0

    # set the density in the logo zones to be really large
    logo_dens = 0.1

    dens[:, :] = logo_dens * (0.5 + logo[0, 0])

    dens.v()[:, :] = (0.5 + logo[:, :]) * logo_dens

    # pressure equilibrium
    gamma = rp.get_param("eos.gamma")

    p_ambient = 1.e-1
    p = myg.scratch_array(nvar=1)
    p[:, :] = p_ambient * (0.8 + logo[0, 0])
    p.v()[:, :] *= (0.8 + logo[:, :])
    # ener[:, :] = p/(gamma - 1.0)
    # ener.v()[:, :] *= (0.2 + logo[:, :])
    rhoh = eos.rhoh_from_rho_p(gamma, dens, p)

    u = xmom/dens
    v = ymom/dens
    W = 1./np.sqrt(1-u**2-v**2)
    dens[:, :] *= W
    xmom[:, :] = rhoh[:, :]*u*W**2
    ymom[:, :] = rhoh[:, :]*v*W**2

    ener[:, :] = rhoh[:, :]*W**2 - p - dens[:, :]


def finalize():
    """ print out any information to the user at the end of the run """

    msg = """
          The script analysis/sedov_compare.py can be used to analyze these
          results.  That will perform an average at constant radius and
          compare the radial profiles to the exact solution.  Sample exact
          data is provided as analysis/cylindrical-sedov.out
          """

    print(msg)
