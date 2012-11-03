from util import runparams

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
    
    u = myData.getVarPtr("x-velocity")
    v = myData.getVarPtr("y-velocity")
    
    # the timestep is min(dx/|u|, dy|v|)
    xtmp = myData.grid.dx/(abs(u))
    ytmp = myData.grid.dy/(abs(v))

    dt = cfl*min(xtmp.min(), ytmp.min())

    return dt



