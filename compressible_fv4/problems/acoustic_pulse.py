from __future__ import print_function

import sys
import mesh.patch as patch
import mesh.fv as fv
import numpy as np
from util import msg
import compressible_fv4.initialization_support as init_support

def init_data(my_data, rp):
    """initialize the acoustic_pulse problem.  This comes from
    McCourquodale & Coella 2011"""

    msg.bold("initializing the acoustic pulse problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, fv.FV2d):
        print("ERROR: patch invalid in acoustic_pulse.py")
        print(my_data.__class__)
        sys.exit()

    # we'll initialize on a finer grid and then average down
    fd = init_support.get_finer(my_data)

    # get the density, momenta, and energy as separate variables
    dens = fd.get_var("density")
    xmom = fd.get_var("x-momentum")
    ymom = fd.get_var("y-momentum")
    ener = fd.get_var("energy")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    xmom[:,:] = 0.0
    ymom[:,:] = 0.0

    gamma = rp.get_param("eos.gamma")

    rho0 = rp.get_param("acoustic_pulse.rho0")
    drho0 = rp.get_param("acoustic_pulse.drho0")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)

    dist = np.sqrt((fd.grid.x2d - xctr)**2 +
                   (fd.grid.y2d - yctr)**2)

    dens[:,:] = rho0
    idx = dist <= 0.5
    dens[idx] = rho0 + drho0*np.exp(-16*dist[idx]**2) * np.cos(np.pi*dist[idx])**6

    p = (dens/rho0)**gamma
    ener[:,:] = p/(gamma - 1)

    init_support.average_down(my_data, fd)


def finalize():
    """ print out any information to the user at the end of the run """
    pass
