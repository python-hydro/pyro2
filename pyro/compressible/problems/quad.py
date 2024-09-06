import numpy as np

from pyro.util import msg

DEFAULT_INPUTS = "inputs.quad"

# these defaults seem to be equivalent to Configuration 3 from
# Shulz-Rinne et al. SIAM J. Sci. Comput., 14, 6, 1394-1414, 1993
#
# Also, with the numbers written out, this is Configuration 3 from
# Lax and Liu, SIAM J. Sci. Comput., 19, 2, 319-340, 1998
#
# See also LeVeque JCP 131, 327-353, 1997

PROBLEM_PARAMS = {"quadrant.rho1": 1.5,  # quadrant 1 initial density
                  "quadrant.u1": 0.0,  # quadrant 1 initial x-velocity
                  "quadrant.v1": 0.0,  # quadrant 1 initial y-velocity
                  "quadrant.p1": 1.5,  # quadrant 1 initial pressure
                  "quadrant.rho2": 0.532258064516129,  # quadrant 2 initial density
                  "quadrant.u2": 1.206045378311055,  # quadrant 2 initial x-velocity
                  "quadrant.v2": 0.0,  # quadrant 2 initial y-velocity
                  "quadrant.p2": 0.3,  # quadrant 2 initial pressure
                  "quadrant.rho3": 0.137992831541219,  # quadrant 3 initial density
                  "quadrant.u3": 1.206045378311055,  # quadrant 3 initial x-velocity
                  "quadrant.v3": 1.206045378311055,  # quadrant 3 initial y-velocity
                  "quadrant.p3": 0.029032258064516,  # quadrant 3 initial pressure
                  "quadrant.rho4": 0.532258064516129,  # quadrant 4 initial density
                  "quadrant.u4": 0.0,  # quadrant 4 initial x-velocity
                  "quadrant.v4": 1.206045378311055,  # quadrant 4 initial y-velocity
                  "quadrant.p4": 0.3,  # quadrant 4 initial pressure
                  "quadrant.cx": 0.5,  # corner x position
                  "quadrant.cy": 0.5}  # corner y position


def init_data(my_data, rp):
    """ initialize the quadrant problem """

    if rp.get_param("driver.verbose"):
        msg.bold("initializing the quadrant problem...")

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    # initialize the components, remember, that ener here is
    # rho*eint + 0.5*rho*v**2, where eint is the specific
    # internal energy (erg/g)
    r1 = rp.get_param("quadrant.rho1")
    u1 = rp.get_param("quadrant.u1")
    v1 = rp.get_param("quadrant.v1")
    p1 = rp.get_param("quadrant.p1")

    r2 = rp.get_param("quadrant.rho2")
    u2 = rp.get_param("quadrant.u2")
    v2 = rp.get_param("quadrant.v2")
    p2 = rp.get_param("quadrant.p2")

    r3 = rp.get_param("quadrant.rho3")
    u3 = rp.get_param("quadrant.u3")
    v3 = rp.get_param("quadrant.v3")
    p3 = rp.get_param("quadrant.p3")

    r4 = rp.get_param("quadrant.rho4")
    u4 = rp.get_param("quadrant.u4")
    v4 = rp.get_param("quadrant.v4")
    p4 = rp.get_param("quadrant.p4")

    cx = rp.get_param("quadrant.cx")
    cy = rp.get_param("quadrant.cy")

    gamma = rp.get_param("eos.gamma")

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
    dens[iq1] = r1
    xmom[iq1] = r1*u1
    ymom[iq1] = r1*v1
    ener[iq1] = p1/(gamma - 1.0) + 0.5*r1*(u1*u1 + v1*v1)

    # quadrant 2
    dens[iq2] = r2
    xmom[iq2] = r2*u2
    ymom[iq2] = r2*v2
    ener[iq2] = p2/(gamma - 1.0) + 0.5*r2*(u2*u2 + v2*v2)

    # quadrant 3
    dens[iq3] = r3
    xmom[iq3] = r3*u3
    ymom[iq3] = r3*v3
    ener[iq3] = p3/(gamma - 1.0) + 0.5*r3*(u3*u3 + v3*v3)

    # quadrant 4
    dens[iq4] = r4
    xmom[iq4] = r4*u4
    ymom[iq4] = r4*v4
    ener[iq4] = p4/(gamma - 1.0) + 0.5*r4*(u4*u4 + v4*v4)


def finalize():
    """ print out any information to the user at the end of the run """
