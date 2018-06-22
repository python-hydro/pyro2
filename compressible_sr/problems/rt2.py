"""A RT problem with two distinct modes: short wave length on the
left and long wavelenght on the right.  This allows one to see
how the growth rate depends on wavenumber.
"""

from __future__ import print_function

import numpy as np

import sys
import mesh.patch as patch
from util import msg


def init_data(my_data, rp):

    """ initialize the rt problem """

    msg.bold("initializing the rt problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in rt2.py")
        print(my_data.__class__)
        sys.exit()

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    gamma = rp.get_param("eos.gamma")

    grav = rp.get_param("compressible.grav")

    dens1 = rp.get_param("rt2.dens1")
    dens2 = rp.get_param("rt2.dens2")
    p0 = rp.get_param("rt2.p0")
    amp = rp.get_param("rt2.amp")
    sigma = rp.get_param("rt2.sigma")

    # initialize the components, remember, that ener here is
    # rho*eint + 0.5*rho*v**2, where eint is the specific
    # internal energy (erg/g)
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0
    dens[:, :] = 0.0

    f_l = 18
    f_r = 3

    # set the density to be stratified in the y-direction
    myg = my_data.grid

    ycenter = 0.5*(myg.ymin + myg.ymax)

    p = myg.scratch_array()

    j = myg.jlo
    while j <= myg.jhi:
        if (myg.y[j] < ycenter):
            dens[:, j] = dens1
            p[:, j] = p0 + dens1*grav*myg.y[j]

        else:
            dens[:, j] = dens2
            p[:, j] = p0 + dens1*grav*ycenter + dens2*grav*(myg.y[j] - ycenter)

        j += 1

    idx_l = myg.x2d < (myg.xmax - myg.xmin)/3.0
    idx_r = myg.x2d >= (myg.xmax - myg.xmin)/3.0

    ymom[idx_l] = amp*np.sin(4.0*np.pi*f_l*myg.x2d[idx_l] /
                             (myg.xmax-myg.xmin))*np.exp(-(myg.y2d[idx_l]-ycenter)**2/sigma**2)
    ymom[idx_r] = amp*np.sin(4.0*np.pi*f_r*myg.x2d[idx_r] /
                             (myg.xmax-myg.xmin))*np.exp(-(myg.y2d[idx_r]-ycenter)**2/sigma**2)

    ymom *= dens

    # set the energy (P = cs2*dens)
    ener[:, :] = p[:, :]/(gamma - 1.0) + \
        0.5*(xmom[:, :]**2 + ymom[:, :]**2)/dens[:, :]


def finalize():
    """ print out any information to the user at the end of the run """
    pass
