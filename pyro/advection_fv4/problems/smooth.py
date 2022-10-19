import numpy

import pyro.mesh.fv as fv
from pyro.util import msg


def init_data(my_data, rp):
    """ initialize the smooth advection problem """

    msg.bold("initializing the smooth FV advection problem...")

    # make sure that we are passed a valid patch object
    # if not isinstance(my_data, patch.FV2d):
    #    print("ERROR: patch invalid in smooth.py")
    #    print(my_data.__class__)
    #    sys.exit()

    xmin = my_data.grid.xmin
    xmax = my_data.grid.xmax

    ymin = my_data.grid.ymin
    ymax = my_data.grid.ymax

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)

    # we need to initialize the cell-averages, so we will create
    # a finer grid, initialize it, and then average down
    mgf = my_data.grid.fine_like(4)

    # since restrict operates in the data class, we need to
    # create a FV2d object here
    fine_data = fv.FV2d(mgf)
    fine_data.register_var("density", my_data.BCs["density"])
    fine_data.create()

    dens_fine = fine_data.get_var("density")

    dens_fine[:, :] = 1.0 + numpy.exp(-60.0*((mgf.x2d-xctr)**2 +
                                             (mgf.y2d-yctr)**2))

    dens = my_data.get_var("density")
    dens[:, :] = fine_data.restrict("density", N=4)


def finalize():
    """ print out any information to the user at the end of the run """
    pass
