from __future__ import print_function

import sys
import numpy as np
import mesh.patch as patch
from util import msg
from sph.particles import Variables, ParticleData2d


def init_data(myd, rp):
    """initialize the circle problem."""

    msg.bold("initializing the circle problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(myd, ParticleData2d):
        print("ERROR: patch invalid in circle.py")
        print(myd.__class__)
        sys.exit()

    U = myd.data
    ivars = Variables(myd)

    h = myd.get_aux("h")
    hh = h / 1.8

    count = 0

    def circ_indicator(x, y):
        dx = x - 0.5
        dy = y - 0.5

        return dx**2 + dy**2 < 0.25**2

    for x in np.arange(0, 1, hh):
        for y in np.arange(0, 1, hh):
            count += int(circ_indicator(x, y))

    p = 0

    for x in np.arange(0, 1, hh):
        for y in np.arange(0, 1, hh):
            if circ_indicator(x, y):
                U[p, ivars.ix] = x
                U[p, ivars.iy] = y
                U[p, ivars.iu:ivars.iv + 1] = 0
                p += 1

    if len(U[:, 0] > p):
        myd.np = p
        myd.data = U[:p, :]

    # give all particles the same mass
    U[:, ivars.im] = 1


def finalize():
    """ print out any information to the user at the end of the run """
    pass
