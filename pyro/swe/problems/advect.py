import numpy as np

from pyro.util import msg

DEFAULT_INPUTS = "inputs.advect"

PROBLEM_PARAMS = {}


def init_data(my_data, rp):
    """ initialize a smooth advection problem for testing convergence """

    if rp.get_param("driver.verbose"):
        msg.bold("initializing the advect problem...")

    # get the hity, momenta, and energy as separate variables
    h = my_data.get_var("height")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    X = my_data.get_var("fuel")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    h[:, :] = 1.0
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)

    # this is identical to the advection/smooth problem
    h[:, :] = 1.0 + np.exp(-60.0*((my_data.grid.x2d-xctr)**2 +
                                    (my_data.grid.y2d-yctr)**2))

    # velocity is diagonal
    u = 1.0
    v = 1.0
    xmom[:, :] = h[:, :]*u
    ymom[:, :] = h[:, :]*v

    X[:, :] = h**2 / np.max(h)


def finalize():
    """ print out any information to the user at the end of the run """

    print("""
          """)
