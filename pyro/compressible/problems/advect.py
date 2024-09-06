import numpy as np

from pyro.util import msg

DEFAULT_INPUTS = "inputs.advect.64"

PROBLEM_PARAMS = {}


def init_data(my_data, rp):
    """ initialize a smooth advection problem for testing convergence """

    if rp.get_param("driver.verbose"):
        msg.bold("initializing the advect problem...")

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    dens[:, :] = 1.0
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0

    gamma = rp.get_param("eos.gamma")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    myg = my_data.grid

    if myg.coord_type == 0:
        xctr = 0.5*(xmin + xmax)
        yctr = 0.5*(ymin + ymax)

        # this is identical to the advection/smooth problem
        dens[:, :] = 1.0 + np.exp(-60.0*((my_data.grid.x2d-xctr)**2 +
                                         (my_data.grid.y2d-yctr)**2))

        # velocity is diagonal
        u = 1.0
        v = 1.0

    else:
        x = myg.scratch_array()
        y = myg.scratch_array()

        xctr = 0.5*(xmin + xmax) * np.sin((ymin + ymax) * 0.25)
        yctr = 0.5*(xmin + xmax) * np.cos((ymin + ymax) * 0.25)

        x[:, :] = myg.x2d.v(buf=myg.ng) * np.sin(myg.y2d.v(buf=myg.ng))
        y[:, :] = myg.x2d.v(buf=myg.ng) * np.cos(myg.y2d.v(buf=myg.ng))

        # this is identical to the advection/smooth problem
        dens[:, :] = 1.0 + np.exp(-120.0*((x-xctr)**2 +
                                         (y-yctr)**2))

        # velocity in theta direction.
        u = 0.0
        v = 1.0

    xmom[:, :] = dens[:, :]*u
    ymom[:, :] = dens[:, :]*v

    # pressure is constant
    p = 1.0
    ener[:, :] = p/(gamma - 1.0) + 0.5*(xmom[:, :]**2 + ymom[:, :]**2)/dens[:, :]


def finalize():
    """ print out any information to the user at the end of the run """

    print("""
          """)
