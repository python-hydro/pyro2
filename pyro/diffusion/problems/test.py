DEFAULT_INPUTS = None

PROBLEM_PARAMS = {}


def init_data(my_data, rp):
    """ an init routine for unit testing """
    del rp  # this problem doesn't use runtime params

    # get the density, momenta, and energy as separate variables
    phi = my_data.get_var("phi")
    phi[:, :] = 1.0


def finalize():
    """ print out any information to the user at the end of the run """
