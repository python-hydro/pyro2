import sys

import numpy as np

from pyro.mesh import patch
from pyro.util import msg


def init_data(my_data, rp):
    """ initialize a smooth advection problem for testing convergence """

    msg.bold("initializing the spherical advect problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in advect.py")
        print(my_data.__class__)
        sys.exit()

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    dens[:, :] = 1.0
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0

    gamma = rp.get_param("eos.gamma")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    # Actual x- and y- coordinate
    myg = my_data.grid

    x = myg.scratch_array()
    y = myg.scratch_array()

    xctr = 0.5*(xmin + xmax) * np.sin((ymin + ymax) * 0.25)
    yctr = 0.5*(xmin + xmax) * np.cos((ymin + ymax) * 0.25)

    x[:, :] = myg.x2d.v(buf=myg.ng) * np.sin(myg.y2d.v(buf=myg.ng))
    y[:, :] = myg.x2d.v(buf=myg.ng) * np.cos(myg.y2d.v(buf=myg.ng))

    # Initial density on the very top.

    # this is identical to the advection/smooth problem
    dens[:, :] = 1.0 + np.exp(-60.0*((x-xctr)**2 +
                                    (y-yctr)**2))

    # velocity in theta direction.
    u = 0.0
    v = 3.0

    xmom[:, :] = dens[:, :]*u
    ymom[:, :] = dens[:, :]*v

    # pressure is constant
    p = 1.0
    ener[:, :] = p/(gamma - 1.0) + 0.5*(xmom[:, :]**2 + ymom[:, :]**2)/dens[:, :]


def finalize():
    """ print out any information to the user at the end of the run """

    print("""
          """)
