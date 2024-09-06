DEFAULT_INPUTS = None

PROBLEM_PARAMS = {}


def init_data(my_data, rp):
    """ an init routine for unit testing """
    del rp  # this problem doesn't use runtime params

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    dens[:, :] = 1.0

    u = my_data.get_var("x-velocity")
    v = my_data.get_var("y-velocity")

    u[:, :] = 1.0
    v[:, :] = 1.0


def finalize():
    """ print out any information to the user at the end of the run """
