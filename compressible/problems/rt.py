from __future__ import print_function

import math
import numpy

import sys
import mesh.patch as patch
from util import msg

def init_data(my_data, rp):
    """ initialize the rt problem """

    msg.bold("initializing the rt problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in rt.py")
        print(my_data.__class__)
        sys.exit()

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    gamma = rp.get_param("eos.gamma")

    grav = rp.get_param("compressible.grav")


    dens1 = rp.get_param("rt.dens1")
    dens2 = rp.get_param("rt.dens2")
    p0 = rp.get_param("rt.p0")
    amp = rp.get_param("rt.amp")
    sigma = rp.get_param("rt.sigma")


    # initialize the components, remember, that ener here is
    # rho*eint + 0.5*rho*v**2, where eint is the specific
    # internal energy (erg/g)
    xmom.d[:,:] = 0.0
    ymom.d[:,:] = 0.0
    dens.d[:,:] = 0.0

    # set the density to be stratified in the y-direction
    myg = my_data.grid

    ycenter = 0.5*(myg.ymin + myg.ymax)

    p = myg.scratch_array()

    j = myg.jlo
    while j <= myg.jhi:
        if (myg.y[j] < ycenter):
            dens.d[:,j] = dens1
            p.d[:,j] = p0 + dens1*grav*myg.y[j]

        else:
            dens.d[:,j] = dens2
            p.d[:,j] = p0 + dens1*grav*ycenter + dens2*grav*(myg.y[j] - ycenter)


        j += 1


    ymom.d[:,:] = amp*numpy.cos(2.0*math.pi*myg.x2d/(myg.xmax-myg.xmin))*numpy.exp(-(myg.y2d-ycenter)**2/sigma**2)

    ymom *= dens

    # set the energy (P = cs2*dens)
    ener.d[:,:] = p.d[:,:]/(gamma - 1.0) + \
        0.5*(xmom.d[:,:]**2 + ymom.d[:,:]**2)/dens.d[:,:]


def finalize():
    """ print out any information to the user at the end of the run """
    pass
