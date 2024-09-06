import numpy as np

from pyro.util import msg

DEFAULT_INPUTS = "inputs.acoustic_pulse"

PROBLEM_PARAMS = {"acoustic_pulse.h0": 1.4,
                  "acoustic_pulse.dh0": 0.14}


def init_data(myd, rp):
    """initialize the acoustic_pulse problem.  This comes from
    McCourquodale & Coella 2011"""

    if rp.get_param("driver.verbose"):
        msg.bold("initializing the acoustic pulse problem...")

    # get the height, momenta as separate variables
    h = myd.get_var("height")
    xmom = myd.get_var("x-momentum")
    ymom = myd.get_var("y-momentum")
    X = myd.get_var("fuel")

    # initialize the components
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0

    h0 = rp.get_param("acoustic_pulse.h0")
    dh0 = rp.get_param("acoustic_pulse.dh0")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)

    dist = np.sqrt((myd.grid.x2d - xctr)**2 +
                   (myd.grid.y2d - yctr)**2)

    h[:, :] = h0
    idx = dist <= 0.5
    h[idx] = h0 + dh0*np.exp(-16*dist[idx]**2) * np.cos(np.pi*dist[idx])**6

    X[:, :] = h[:, :]**2 / np.max(h)


def finalize():
    """ print out any information to the user at the end of the run """
