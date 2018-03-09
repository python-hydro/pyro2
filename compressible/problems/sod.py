from __future__ import print_function

import sys

import mesh.patch as patch
from util import msg


def init_data(my_data, rp):
    """ initialize the sod problem """

    msg.bold("initializing the sod problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in sod.py")
        print(my_data.__class__)
        sys.exit()

    # get the sod parameters
    dens_left = rp.get_param("sod.dens_left")
    dens_right = rp.get_param("sod.dens_right")

    u_left = rp.get_param("sod.u_left")
    u_right = rp.get_param("sod.u_right")

    p_left = rp.get_param("sod.p_left")
    p_right = rp.get_param("sod.p_right")

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    gamma = rp.get_param("eos.gamma")

    direction = rp.get_param("sod.direction")

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)

    myg = my_data.grid

    if direction == "x":

        # left
        idxl = myg.x2d <= xctr

        dens[idxl] = dens_left
        xmom[idxl] = dens_left*u_left
        ymom[idxl] = 0.0
        ener[idxl] = p_left/(gamma - 1.0) + 0.5*xmom[idxl]*u_left

        # right
        idxr = myg.x2d > xctr

        dens[idxr] = dens_right
        xmom[idxr] = dens_right*u_right
        ymom[idxr] = 0.0
        ener[idxr] = p_right/(gamma - 1.0) + 0.5*xmom[idxr]*u_right

    else:

        # bottom
        idxb = myg.y2d <= yctr

        dens[idxb] = dens_left
        xmom[idxb] = 0.0
        ymom[idxb] = dens_left*u_left
        ener[idxb] = p_left/(gamma - 1.0) + 0.5*ymom[idxb]*u_left

        # top
        idxt = myg.y2d > yctr

        dens[idxt] = dens_right
        xmom[idxt] = 0.0
        ymom[idxt] = dens_right*u_right
        ener[idxt] = p_right/(gamma - 1.0) + 0.5*ymom[idxt]*u_right


def finalize():
    """ print out any information to the user at the end of the run """

    msg = """
          The script analysis/sod_compare.py can be used to compare
          this output to the exact solution.  Some sample exact solution
          data is present as analysis/sod-exact.out
          """

    print(msg)
