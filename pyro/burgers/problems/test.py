from pyro.util import msg

DEFAULT_INPUTS = "inputs.test"

PROBLEM_PARAMS = {}


def init_data(myd, rp):
    """ initialize the burgers test problem """

    if rp.get_param("driver.verbose"):
        msg.bold("initializing the burgers test problem...")

    u = myd.get_var("x-velocity")
    v = myd.get_var("y-velocity")

    u[:, :] = 3.0
    v[:, :] = 3.0

    # y = -x + 1

    index = myd.grid.y2d > -1.0 * myd.grid.x2d + 1.0

    u[index] = 1.0
    v[index] = 1.0


def finalize():
    """ print out any information to the user at the end of the run """
