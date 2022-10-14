import sys

import numpy

import pyro.mesh.patch as patch
from pyro.util import msg


def phi_analytic(dist, t, t_0, k, phi_1, phi_2):
    """ the analytic solution to the Gaussian diffusion problem """
    phi = (phi_2 - phi_1)*(t_0/(t + t_0)) * \
        numpy.exp(-0.25*dist**2/(k*(t + t_0))) + phi_1
    return phi


def init_data(my_data, rp):
    """ initialize the Gaussian diffusion problem """

    msg.bold("initializing the Gaussian diffusion problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in diffuse.py")
        print(my_data.__class__)
        sys.exit()

    phi = my_data.get_var("phi")

    xmin = my_data.grid.xmin
    xmax = my_data.grid.xmax

    ymin = my_data.grid.ymin
    ymax = my_data.grid.ymax

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)

    k = rp.get_param("diffusion.k")
    t_0 = rp.get_param("gaussian.t_0")
    phi_max = rp.get_param("gaussian.phi_max")
    phi_0 = rp.get_param("gaussian.phi_0")

    dist = numpy.sqrt((my_data.grid.x2d - xctr)**2 +
                      (my_data.grid.y2d - yctr)**2)

    phi[:, :] = phi_analytic(dist, 0.0, t_0, k, phi_0, phi_max)

    # for later interpretation / analysis, store some auxillary data
    my_data.set_aux("k", k)
    my_data.set_aux("t_0", t_0)
    my_data.set_aux("phi_0", phi_0)
    my_data.set_aux("phi_max", phi_max)


def finalize():
    """ print out any information to the user at the end of the run """

    ostr = """
          The solution can be compared to the analytic solution with
          the script analysis/gauss_diffusion_compare.py
          """

    print(ostr)
