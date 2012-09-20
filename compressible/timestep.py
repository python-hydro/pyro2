from util import runparams
import eos
import numpy


"""
The timestep module computes the advective timestep (CFL) constraint.
The CFL constraint says that information cannot propagate further than
one zone per timestep.

We use the driver.cfl parameter to control what fraction of the CFL
step we actually take.
"""

SMALL = 1.e-12

def timestep(myData):
    """ 
    compute the CFL timestep for the current patch.
    """

    cfl = runparams.getParam("driver.cfl")


    # get the variables we need                                                 
    dens = myData.getVarPtr("density")
    xmom = myData.getVarPtr("x-momentum")
    ymom = myData.getVarPtr("y-momentum")
    ener = myData.getVarPtr("energy")


    # we need to compute the pressure
    u = xmom/dens
    v = ymom/dens

    e = (ener - 0.5*dens*(u*u + v*v))/dens

    p = eos.pres(dens, e)

    # compute the sounds speed
    gamma = runparams.getParam("eos.gamma")

    cs = numpy.sqrt(gamma*p/dens)


    # the timestep is min(dx/(|u| + cs), dy/(|v| + cs))
    xtmp = myData.grid.dx/(abs(u) + cs)
    ytmp = myData.grid.dy/(abs(v) + cs)

    dt = cfl*min(xtmp.min(), ytmp.min())

    return dt
