from __future__ import print_function

import sys

import numpy as np

import mesh.patch as patch
from util import msg


def init_data(my_data, rp):
    """ initialize the quadrant problem """

    msg.bold("initializing the quadrant problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in quad.py")
        print(my_data.__class__)
        sys.exit()

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    # initialize the components, remember, that ener here is
    # rho*eint + 0.5*rho*v**2, where eint is the specific
    # internal energy (erg/g)
    r1 = rp.get_param("quadrant.rho1")
    u1 = rp.get_param("quadrant.u1")
    v1 = rp.get_param("quadrant.v1")
    p1 = rp.get_param("quadrant.p1")

    r2 = rp.get_param("quadrant.rho2")
    u2 = rp.get_param("quadrant.u2")
    v2 = rp.get_param("quadrant.v2")
    p2 = rp.get_param("quadrant.p2")

    r3 = rp.get_param("quadrant.rho3")
    u3 = rp.get_param("quadrant.u3")
    v3 = rp.get_param("quadrant.v3")
    p3 = rp.get_param("quadrant.p3")

    r4 = rp.get_param("quadrant.rho4")
    u4 = rp.get_param("quadrant.u4")
    v4 = rp.get_param("quadrant.v4")
    p4 = rp.get_param("quadrant.p4")

    cx = rp.get_param("quadrant.cx")
    cy = rp.get_param("quadrant.cy")

    gamma = rp.get_param("eos.gamma")

    # there is probably an easier way to do this, but for now, we
    # will just do an explicit loop.  Also, we really want to set
    # the pressue and get the internal energy from that, and then
    # compute the total energy (which is what we store).  For now
    # we will just fake this

    myg = my_data.grid

    iq1 = np.logical_and(myg.x2d >= cx, myg.y2d >= cy)
    iq2 = np.logical_and(myg.x2d < cx,  myg.y2d >= cy)
    iq3 = np.logical_and(myg.x2d < cx,  myg.y2d < cy)
    iq4 = np.logical_and(myg.x2d >= cx, myg.y2d < cy)

    # quadrant 1
    dens[iq1] = r1
    xmom[iq1] = r1*u1
    ymom[iq1] = r1*v1
    ener[iq1] = p1/(gamma - 1.0) + 0.5*r1*(u1*u1 + v1*v1)

    # quadrant 2
    dens[iq2] = r2
    xmom[iq2] = r2*u2
    ymom[iq2] = r2*v2
    ener[iq2] = p2/(gamma - 1.0) + 0.5*r2*(u2*u2 + v2*v2)

    # quadrant 3
    dens[iq3] = r3
    xmom[iq3] = r3*u3
    ymom[iq3] = r3*v3
    ener[iq3] = p3/(gamma - 1.0) + 0.5*r3*(u3*u3 + v3*v3)

    # quadrant 4
    dens[iq4] = r4
    xmom[iq4] = r4*u4
    ymom[iq4] = r4*v4
    ener[iq4] = p4/(gamma - 1.0) + 0.5*r4*(u4*u4 + v4*v4)


def finalize():
    """ print out any information to the user at the end of the run """
    pass
