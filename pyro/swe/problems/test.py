DEFAULT_INPTUS = None

PROBLEM_PARAMS = {}


def init_data(my_data, rp):
    """ an init routine for unit testing """
    del rp  # this problem doesn't use runtime params

    # get the hity, momenta, and energy as separate variables
    h = my_data.get_var("height")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")

    # initialize the components
    h[:, :] = 1.0
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0


def finalize():
    """ print out any information to the user at the end of the run """
