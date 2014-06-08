from util import runparams

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

    cfl = runparams.getParam("driver.cfl")
    
    u = runparams.getParam("advection.u")
    v = runparams.getParam("advection.v")
    
    # the timestep is min(dx/|u|, dy|v|)
    xtmp = my_data.grid.dx/max(abs(u),SMALL)
    ytmp = my_data.grid.dy/max(abs(v),SMALL)

    dt = cfl*min(xtmp, ytmp)

    return dt



