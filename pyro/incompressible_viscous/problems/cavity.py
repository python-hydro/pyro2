r"""
Initialize the lid-driven cavity problem. Run on a unit square with the top wall
moving to the right with unit velocity, driving the flow. The other three walls
are no-slip boundary conditions. The initial velocity of the fluid is zero.

Since the length and velocity scales are both 1, the Reynolds number is 1/viscosity.

References:
https://doi.org/10.1007/978-3-319-91494-7_8
https://www.fluid.tuwien.ac.at/HendrikKuhlmann?action=AttachFile&do=get&target=LidDrivenCavity.pdf
"""

from pyro.util import msg

DEFAULT_INPUTS = "inputs.cavity"

PROBLEM_PARAMS = {}


def init_data(my_data, rp):
    """ initialize the lid-driven cavity """

    if rp.get_param("driver.verbose"):
        msg.bold("initializing the lid-driven cavity problem...")

    myg = my_data.grid

    if (myg.xmin != 0 or myg.xmax != 1 or
        myg.ymin != 0 or myg.ymax != 1):
        msg.fail("ERROR: domain should be a unit square")

    # get the velocities and set them to zero
    u = my_data.get_var("x-velocity")
    v = my_data.get_var("y-velocity")
    u[:, :] = 0
    v[:, :] = 0


def finalize():
    """ print out any information to the user at the end of the run """
