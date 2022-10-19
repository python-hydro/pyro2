r"""
Initialize the doubly periodic shear layer (see, for example, Martin
and Colella, 2000, JCP, 163, 271).  This is run in a unit square
domain, with periodic boundary conditions on all sides.  Here, the
initial velocity is::

                 / tanh(rho_s (y-0.25))   if y <= 0.5
   u(x,y,t=0) = <
                 \ tanh(rho_s (0.75-y))   if y > 0.5

   v(x,y,t=0) = delta_s sin(2 pi x)


"""


import math

import numpy as np

import pyro.mesh.patch as patch
from pyro.util import msg


def init_data(my_data, rp):
    """ initialize the incompressible shear problem """

    msg.bold("initializing the incompressible shear problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print(my_data.__class__)
        msg.fail("ERROR: patch invalid in shear.py")

    # get the necessary runtime parameters
    rho_s = rp.get_param("shear.rho_s")
    delta_s = rp.get_param("shear.delta_s")

    # get the velocities
    u = my_data.get_var("x-velocity")
    v = my_data.get_var("y-velocity")

    myg = my_data.grid

    if (myg.xmin != 0 or myg.xmax != 1 or
        myg.ymin != 0 or myg.ymax != 1):
        msg.fail("ERROR: domain should be a unit square")

    y_half = 0.5*(myg.ymin + myg.ymax)

    print('y_half = ', y_half)
    print('delta_s = ', delta_s)
    print('rho_s = ', rho_s)

    idx = myg.y2d <= y_half
    u[idx] = np.tanh(rho_s*(myg.y2d[idx] - 0.25))

    idx = myg.y2d > y_half
    u[idx] = np.tanh(rho_s*(0.75 - myg.y2d[idx]))

    v[:, :] = delta_s*np.sin(2.0*math.pi*myg.x2d)

    print("extrema: ", u.min(), u.max())


def finalize():
    """ print out any information to the user at the end of the run """
    pass
