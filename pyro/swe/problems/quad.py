import numpy as np

from pyro.util import msg

DEFAULT_INPUTS = "inputs.quad"

PROBLEM_PARAMS = {"quadrant.h1": 1.5,  # quadrant 1 initial density
                  "quadrant.u1": 0.0,  # quadrant 1 initial x-velocity
                  "quadrant.v1": 0.0,  # quadrant 1 initial y-velocity
                  "quadrant.h2": 0.532258064516129,  # quadrant 2 initial density
                  "quadrant.u2": 1.206045378311055,  # quadrant 2 initial x-velocity
                  "quadrant.v2": 0.0,  # quadrant 2 initial y-velocity
                  "quadrant.h3": 0.137992831541219,  # quadrant 3 initial density
                  "quadrant.u3": 1.206045378311055,  # quadrant 3 initial x-velocity
                  "quadrant.v3": 1.206045378311055,  # quadrant 3 initial y-velocity
                  "quadrant.h4": 0.532258064516129,  # quadrant 4 initial density
                  "quadrant.u4": 0.0,  # quadrant 4 initial x-velocity
                  "quadrant.v4": 1.206045378311055,  # quadrant 4 initial y-velocity
                  "quadrant.cx": 0.5,  # corner x position
                  "quadrant.cy": 0.5}  # corner y position


def init_data(my_data, rp):
    """ initialize the quadrant problem """

    if rp.get_param("driver.verbose"):
        msg.bold("initializing the quadrant problem...")

    # get the height, momenta, and energy as separate variables
    h = my_data.get_var("height")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    X = my_data.get_var("fuel")

    # initialize the components, remember, that ener here is
    # h*eint + 0.5*h*v**2, where eint is the specific
    # internal energy (erg/g)
    r1 = rp.get_param("quadrant.h1")
    u1 = rp.get_param("quadrant.u1")
    v1 = rp.get_param("quadrant.v1")

    r2 = rp.get_param("quadrant.h2")
    u2 = rp.get_param("quadrant.u2")
    v2 = rp.get_param("quadrant.v2")

    r3 = rp.get_param("quadrant.h3")
    u3 = rp.get_param("quadrant.u3")
    v3 = rp.get_param("quadrant.v3")

    r4 = rp.get_param("quadrant.h4")
    u4 = rp.get_param("quadrant.u4")
    v4 = rp.get_param("quadrant.v4")

    cx = rp.get_param("quadrant.cx")
    cy = rp.get_param("quadrant.cy")

    # there is probably an easier way to do this, but for now, we
    # will just do an explicit loop.  Also, we really want to set
    # the pressure and get the internal energy from that, and then
    # compute the total energy (which is what we store).  For now
    # we will just fake this

    myg = my_data.grid

    iq1 = np.logical_and(myg.x2d >= cx, myg.y2d >= cy)
    iq2 = np.logical_and(myg.x2d < cx,  myg.y2d >= cy)
    iq3 = np.logical_and(myg.x2d < cx,  myg.y2d < cy)
    iq4 = np.logical_and(myg.x2d >= cx, myg.y2d < cy)

    # quadrant 1
    h[iq1] = r1
    xmom[iq1] = r1*u1
    ymom[iq1] = r1*v1
    X[iq1] = 1.0

    # quadrant 2
    h[iq2] = r2
    xmom[iq2] = r2*u2
    ymom[iq2] = r2*v2
    X[iq2] = 0.0

    # quadrant 3
    h[iq3] = r3
    xmom[iq3] = r3*u3
    ymom[iq3] = r3*v3
    X[iq3] = 1.0

    # quadrant 4
    h[iq4] = r4
    xmom[iq4] = r4*u4
    ymom[iq4] = r4*v4
    X[iq4] = 0.0

    X *= h


def finalize():
    """ print out any information to the user at the end of the run """
