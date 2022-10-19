import sys

import pyro.compressible_sr.eos as eos
import pyro.mesh.patch as patch


def init_data(my_data, rp):
    """ an init routine for unit testing """

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in sedov.py")
        print(my_data.__class__)
        sys.exit()

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    gamma = rp.get_param("eos.gamma")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    dens[:, :] = 1.0
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0
    # ener[:, :] = 2.5

    p = 1.0

    rhoh = eos.rhoh_from_rho_p(gamma, dens, p)
    # print(f'rhoh = {rhoh}')

    # u, v = 0 so W = 1

    ener[:, :] = rhoh[:, :] - p - dens[:, :]


def finalize():
    """ print out any information to the user at the end of the run """
    pass
