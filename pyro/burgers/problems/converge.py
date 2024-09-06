import numpy

from pyro.util import msg

DEFAULT_INPUTS = "inputs.converge.64"


def init_data(my_data, rp):
    """ initialize the smooth burgers convergence problem """

    if rp.get_param("driver.verbose"):
        msg.bold("initializing the smooth burgers convergence problem...")

    u = my_data.get_var("x-velocity")
    v = my_data.get_var("y-velocity")

    xmin = my_data.grid.xmin
    xmax = my_data.grid.xmax

    ymin = my_data.grid.ymin
    ymax = my_data.grid.ymax

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)

    # A represents some magnitude that defines the initial u and v.

    A = 0.05

    u[:, :] = A + A * numpy.exp(-50.0*((my_data.grid.x2d-xctr)**2 +
                                        (my_data.grid.y2d-yctr)**2))
    v[:, :] = A + A * numpy.exp(-50.0*((my_data.grid.x2d-xctr)**2 +
                                        (my_data.grid.y2d-yctr)**2))


def finalize():
    """ print out any information to the user at the end of the run """
