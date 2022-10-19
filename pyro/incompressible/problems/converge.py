r"""
Initialize a smooth incompressible convergence test.  Here, the
velocities are initialized as

.. math::

    u(x,y) = 1 - 2 \cos(2 \pi x) \sin(2 \pi y)

    v(x,y) = 1 + 2 \sin(2 \pi x) \cos(2 \pi y)

and the exact solution at some later time t is then

.. math::

    u(x,y,t) = 1 - 2 \cos(2 \pi (x - t)) \sin(2 \pi (y - t))

    v(x,y,t) = 1 + 2 \sin(2 \pi (x - t)) \cos(2 \pi (y - t))

    p(x,y,t) = -\cos(4 \pi (x - t)) - \cos(4 \pi (y - t))

The numerical solution can be compared to the exact solution to
measure the convergence rate of the algorithm.

"""


import math

import numpy as np

import pyro.mesh.patch as patch
from pyro.util import msg


def init_data(my_data, rp):
    """ initialize the incompressible converge problem """

    msg.bold("initializing the incompressible converge problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print(my_data.__class__)
        msg.fail("ERROR: patch invalid in converge.py")

    # get the velocities
    u = my_data.get_var("x-velocity")
    v = my_data.get_var("y-velocity")

    myg = my_data.grid

    if (myg.xmin != 0 or myg.xmax != 1 or
        myg.ymin != 0 or myg.ymax != 1):
        msg.fail("ERROR: domain should be a unit square")

    u[:, :] = 1.0 - 2.0*np.cos(2.0*math.pi*myg.x2d)*np.sin(2.0*math.pi*myg.y2d)
    v[:, :] = 1.0 + 2.0*np.sin(2.0*math.pi*myg.x2d)*np.cos(2.0*math.pi*myg.y2d)


def finalize():
    """ print out any information to the user at the end of the run """

    ostr = """
          Comparisons to the analytic solution can be done using
          analysis/incomp_converge_error.py
          """

    print(ostr)
