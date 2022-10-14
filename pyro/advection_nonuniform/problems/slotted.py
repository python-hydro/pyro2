import numpy as np

import pyro.mesh.patch as patch
from pyro.util import msg


def init_data(my_data, rp):
    """ initialize the slotted advection problem """
    msg.bold("initializing the slotted advection problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print(my_data.__class__)
        msg.fail("ERROR: patch invalid in slotted.py")

    offset = rp.get_param("slotted.offset")
    omega = rp.get_param("slotted.omega")

    myg = my_data.grid

    xctr_dens = 0.5*(myg.xmin + myg.xmax)
    yctr_dens = 0.5*(myg.ymin + myg.ymax) + offset

    # setting initial condition for density
    dens = my_data.get_var("density")
    dens[:, :] = 0.0

    R = 0.15
    slot_width = 0.05

    inside = (myg.x2d - xctr_dens)**2 + (myg.y2d - yctr_dens)**2 < R**2

    slot_x = np.logical_and(myg.x2d > (xctr_dens - slot_width*0.5),
                            myg.x2d < (xctr_dens + slot_width*0.5))
    slot_y = np.logical_and(myg.y2d > (yctr_dens - R),
                            myg.y2d < (yctr_dens))
    slot = np.logical_and(slot_x, slot_y)

    dens[inside] = 1.0
    dens[slot] = 0.0

    # setting initial condition for velocity
    u = my_data.get_var("x-velocity")
    v = my_data.get_var("y-velocity")

    u[:, :] = omega*(myg.y2d - xctr_dens)
    v[:, :] = -omega*(myg.x2d - (yctr_dens-offset))

    print("extrema: ", np.amax(u), np.amin(u))


def finalize():
    """ print out any information to the user at the end of the run """
    pass
