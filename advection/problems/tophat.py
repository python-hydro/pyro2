from __future__ import print_function

import sys
import mesh.patch as patch
import numpy
from util import msg

def init_data(my_data, rp):
    """ initialize the tophat advection problem """

    msg.bold("initializing the tophat advection problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in tophat.py")
        print(my_data.__class__)
        sys.exit()

    dens = my_data.get_var("density")

    xmin = my_data.grid.xmin
    xmax = my_data.grid.xmax

    ymin = my_data.grid.ymin
    ymax = my_data.grid.ymax

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)

    dens[:,:] = 0.0

    for i in range(my_data.grid.ilo, my_data.grid.ihi+1):
        for j in range(my_data.grid.jlo, my_data.grid.jhi+1):

            if (numpy.sqrt((my_data.grid.x[i]-xctr)**2 +
                           (my_data.grid.y[j]-yctr)**2) < 0.1):
                dens[i,j] = 1.0


def finalize():
    """ print out any information to the user at the end of the run """
    pass

