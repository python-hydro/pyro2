from __future__ import print_function

import sys
import mesh.patch as patch
import numpy
from util import msg


def init_data(my_data, rp):
    """ initialize the Gresho vortex problem """

    msg.bold("initializing the Gresho vortex problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in bubble.py")
        print(my_data.__class__)
        sys.exit()

    # get the density and velocities
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    grav = rp.get_param("compressible.grav")

    gamma = rp.get_param("eos.gamma")

    scale_height = rp.get_param("gresho.scale_height")
    dens_base = rp.get_param("gresho.dens_base")
    dens_cutoff = rp.get_param("gresho.dens_cutoff")

    R = rp.get_param("gresho.r")
    u0 = rp.get_param("gresho.u0")
    p0 = rp.get_param("gresho.p0")

    # initialize the components -- we'll get a psure too
    # but that is used only to initialize the base state
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0
    dens[:, :] = dens_cutoff

    # set the density to be stratified in the y-direction
    myg = my_data.grid
    p = myg.scratch_array()

    dens[:, :] = dens_base

    # for j in range(myg.jlo, myg.jhi+1):
    #     dens[:, j] = max(dens_base*numpy.exp(-myg.y[j]/scale_height),
    #                      dens_cutoff)

    # cs2 = scale_height*abs(grav)

    # set the psure (P = cs2*dens)
    p[:,:] = p0#cs2*dens

    x_centre = 0.5 * (myg.x[0] + myg.x[-1])
    y_centre = 0.5 * (myg.y[0] + myg.y[-1])

    r = numpy.sqrt((myg.x2d - x_centre)**2 + (myg.y2d - y_centre)**2)

    p[r <= R] += 0.5 * (u0 * r[r<=R]/R)**2
    p[(r > R) & (r <= 2*R)] += u0**2 * (0.5 *(r[(r > R) & (r <= 2*R)]/R)**2 + 4 * (1 - r[(r > R) & (r <= 2*R)]/R + numpy.log(r[(r > R) & (r <= 2*R)]/R)))
    p[r > 2*R] += u0**2 * (4 * numpy.log(2) - 2)
    #
    uphi = numpy.zeros_like(p)
    uphi[r <= R] = u0 * r[r<=R]/R
    uphi[(r > R) & (r <= 2*R)] = u0 * (2 - r[(r > R) & (r <= 2*R)]/R)

    xmom[:,:] = -dens[:,:] * uphi[:,:] * (myg.y2d - y_centre) / r[:,:]
    ymom[:,:] = dens[:,:] * uphi[:,:] * (myg.x2d - x_centre) / r[:,:]

    ener[:, :] = p[:, :]/(gamma - 1.0) + \
                0.5*(xmom[:, :]**2 + ymom[:, :]**2)/dens[:, :]

    eint = p[:, :]/(gamma - 1.0)

    dens[:,:] = p[:,:]/(eint[:,:]*(gamma - 1.0))

    print(p)



def finalize():
    """ print out any information to the user at the end of the run """
    pass
