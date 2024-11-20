"""Initialize a tophat profile---the value inside a small circular
regions is set to 1.0 and is zero otherwise.  This will exercise the
limiters significantly.

"""

from pyro.util import msg

DEFAULT_INPUTS = "inputs.tophat"

PROBLEM_PARAMS = {}


def init_data(myd, rp):
    """ initialize the tophat advection problem """

    if rp.get_param("driver.verbose"):
        msg.bold("initializing the tophat advection problem...")

    dens = myd.get_var("density")

    xmin = myd.grid.xmin
    xmax = myd.grid.xmax

    ymin = myd.grid.ymin
    ymax = myd.grid.ymax

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)

    dens[:, :] = 0.0

    R = 0.1

    inside = (myd.grid.x2d - xctr)**2 + (myd.grid.y2d - yctr)**2 < R**2

    dens[inside] = 1.0


def finalize():
    """ print out any information to the user at the end of the run """
