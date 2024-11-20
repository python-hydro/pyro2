r"""
Initialize the doubly periodic shear layer (see, for example, Martin
and Colella, 2000, JCP, 163, 271).  This is run in a unit square
domain, with periodic boundary conditions on all sides.  Here, the
initial velocity is:

.. math::

   u(x,y,t=0) = \begin{cases}
                \tanh(\rho_s (y - 1/4)) &  \mbox{if}~ y \le 1/2 \\
                \tanh(\rho_s (3/4 - y)) &  \mbox{if}~ y > 1/2
                \end{cases}

.. math::

   v(x,y,t=0) = \delta_s \sin(2 \pi x)
"""


import math

import numpy as np

from pyro.util import msg

DEFAULT_INPUTS = "inputs.shear"

PROBLEM_PARAMS = {"shear.rho_s": 42.0,  # shear layer width
                  "shear.delta_s": 0.05}  # perturbuation amplitude


def init_data(my_data, rp):
    """ initialize the incompressible shear problem """

    if rp.get_param("driver.verbose"):
        msg.bold("initializing the incompressible shear problem...")

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
