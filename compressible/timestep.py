import numpy

import eos

"""
The timestep module computes the advective timestep (CFL) constraint.
The CFL constraint says that information cannot propagate further than
one zone per timestep.

We use the driver.cfl parameter to control what fraction of the CFL
step we actually take.
"""

SMALL = 1.e-12

def timestep(my_data):
    """ 
    compute the CFL timestep for the current patch.
    """

    cfl = my_data.rp.get_param("driver.cfl")


    # get the variables we need                                                 
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")


    # we need to compute the pressure
    u = xmom/dens
    v = ymom/dens

    e = (ener - 0.5*dens*(u*u + v*v))/dens

    p = eos.pres(dens, e)

    # compute the sounds speed
    gamma = my_data.rp.get_param("eos.gamma")

    cs = numpy.sqrt(gamma*p/dens)


    # the timestep is min(dx/(|u| + cs), dy/(|v| + cs))
    xtmp = my_data.grid.dx/(abs(u) + cs)
    ytmp = my_data.grid.dy/(abs(v) + cs)

    dt = cfl*min(xtmp.min(), ytmp.min())

    return dt
