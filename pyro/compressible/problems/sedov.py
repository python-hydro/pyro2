import math

import numpy as np

from pyro.util import msg

DEFAULT_INPUTS = "inputs.sedov"

PROBLEM_PARAMS = {"sedov.r_init": 0.1,  # radius for the initial perturbation
                  "sedov.nsub": 4}


def init_data(my_data, rp):
    """ initialize the sedov problem """

    if rp.get_param("driver.verbose"):
        msg.bold("initializing the sedov problem...")

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

    r_init = rp.get_param("sedov.r_init")

    gamma = rp.get_param("eos.gamma")
    pi = math.pi

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    grid = my_data.grid

    if grid.coord_type == 0:
        # If we do Cartesian2d geometry

        E_sedov = 1.0

        xctr = 0.5*(xmin + xmax)
        yctr = 0.5*(ymin + ymax)

        # initialize the pressure by putting the explosion energy into a
        # volume of constant pressure.  Then compute the energy in a zone
        # from this.
        nsub = rp.get_param("sedov.nsub")

        dist = np.sqrt((my_data.grid.x2d - xctr)**2 +
                       (my_data.grid.y2d - yctr)**2)

        p = 1.e-5
        ener[:, :] = p/(gamma - 1.0)

        for i, j in np.transpose(np.nonzero(dist < 2.0*r_init)):

            xsub = my_data.grid.xl[i] + (my_data.grid.dx/nsub)*(np.arange(nsub) + 0.5)
            ysub = my_data.grid.yl[j] + (my_data.grid.dy/nsub)*(np.arange(nsub) + 0.5)

            xx, yy = np.meshgrid(xsub, ysub, indexing="ij")

            dist = np.sqrt((xx - xctr)**2 + (yy - yctr)**2)

            n_in_pert = np.count_nonzero(dist <= r_init)

            p = n_in_pert*(gamma - 1.0)*E_sedov/(pi*r_init*r_init) + \
                (nsub*nsub - n_in_pert)*1.e-5

            p = p/(nsub*nsub)

            ener[i, j] = p/(gamma - 1.0)

    else:
        # If we do SphericalPolar geometry

        # Just put a high energy for now.
        E_sedov = 1.e6

        p = 1.e-6
        ener[:, :] = p/(gamma - 1.0)
        myg = my_data.grid
        ener[myg.x2d < r_init] = E_sedov


def finalize():
    """ print out any information to the user at the end of the run """

    print("""
          The script analysis/sedov_compare.py can be used to analyze these
          results.  That will perform an average at constant radius and
          compare the radial profiles to the exact solution.  Sample exact
          data is provided as analysis/cylindrical-sedov.out
          """)
