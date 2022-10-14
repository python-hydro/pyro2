import sys

import numpy

import pyro.mesh.patch as patch
from pyro.util import msg


def init_data(my_data, base, rp):
    """ initialize the Gresho vortex problem """

    msg.bold("initializing the Gresho vortex problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in bubble.py")
        print(my_data.__class__)
        sys.exit()

    # get the density and velocities
    dens = my_data.get_var("density")
    xvel = my_data.get_var("x-velocity")
    yvel = my_data.get_var("y-velocity")
    eint = my_data.get_var("eint")

    grav = rp.get_param("lm-atmosphere.grav")

    gamma = rp.get_param("eos.gamma")

    scale_height = rp.get_param("gresho.scale_height")
    dens_base = rp.get_param("gresho.dens_base")
    dens_cutoff = rp.get_param("gresho.dens_cutoff")

    R = rp.get_param("gresho.r")
    u0 = rp.get_param("gresho.u0")

    # initialize the components -- we'll get a pressure too
    # but that is used only to initialize the base state
    xvel[:, :] = 0.0
    yvel[:, :] = 0.0
    dens[:, :] = dens_cutoff

    # set the density to be stratified in the y-direction
    myg = my_data.grid
    pres = myg.scratch_array()

    for j in range(myg.jlo, myg.jhi+1):
        dens[:, j] = max(dens_base*numpy.exp(-myg.y[j]/scale_height),
                         dens_cutoff)

    cs2 = scale_height*abs(grav)

    # set the pressure (P = cs2*dens)
    pres = cs2*dens
    eint[:, :] = pres/(gamma - 1.0)/dens

    x_centre = 0.5 * (myg.x[0] + myg.x[-1])
    y_centre = 0.5 * (myg.y[0] + myg.y[-1])

    r = numpy.sqrt((myg.x2d - x_centre)**2 + (myg.y2d - y_centre)**2)

    pres[r <= R] += 0.5 * (u0 * r[r <= R]/R)**2
    pres[(r > R) & (r <= 2*R)] += u0**2 * \
        (0.5 * (r[(r > R) & (r <= 2*R)]/R)**2 +
        4 * (1 - r[(r > R) & (r <= 2*R)]/R +
        numpy.log(r[(r > R) & (r <= 2*R)]/R)))
    pres[r > 2*R] += u0**2 * (4 * numpy.log(2) - 2)
    #
    uphi = numpy.zeros_like(pres)
    uphi[r <= R] = u0 * r[r <= R]/R
    uphi[(r > R) & (r <= 2*R)] = u0 * (2 - r[(r > R) & (r <= 2*R)]/R)

    xvel[:, :] = -uphi[:, :] * (myg.y2d - y_centre) / r[:, :]
    yvel[:, :] = uphi[:, :] * (myg.x2d - x_centre) / r[:, :]

    dens[:, :] = pres[:, :]/(eint[:, :]*(gamma - 1.0))

    # do the base state
    base["rho0"].d[:] = numpy.mean(dens, axis=0)
    base["p0"].d[:] = numpy.mean(pres, axis=0)

    # redo the pressure via HSE
    for j in range(myg.jlo+1, myg.jhi):
        base["p0"].d[j] = base["p0"].d[j-1] + 0.5*myg.dy*(base["rho0"].d[j] + base["rho0"].d[j-1])*grav


def finalize():
    """ print out any information to the user at the end of the run """
    pass
