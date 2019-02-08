from __future__ import print_function

import pyro2.mesh.patch as patch
from pyro2.util import msg


def init_data(my_data, rp):
    """ initialize the tophat advection problem """

    msg.bold("initializing the tophat advection problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print(my_data.__class__)
        msg.fail("ERROR: patch invalid in slotted.py")

    dens = my_data.get_var("density")

    xmin = my_data.grid.xmin
    xmax = my_data.grid.xmax

    ymin = my_data.grid.ymin
    ymax = my_data.grid.ymax

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)

    dens[:, :] = 0.0

    R = 0.1

    inside = (my_data.grid.x2d - xctr)**2 + (my_data.grid.y2d - yctr)**2 < R**2

    dens[inside] = 1.0

    u = my_data.get_var("x-velocity")
    v = my_data.get_var("y-velocity")

    u[:, :] = rp.get_param("advection.u")
    v[:, :] = rp.get_param("advection.v")


def finalize():
    """ print out any information to the user at the end of the run """
    pass
