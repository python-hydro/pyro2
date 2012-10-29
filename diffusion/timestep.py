from util import runparams

"""
The diffusion timestep module computes the timestep using the explicit
timestep constraint as the starting point.  We then
"""

def timestep(myData):
    """ 
    compute the CFL timestep for the current patch.
    """

    cfl = runparams.getParam("driver.cfl")
    
    u = runparams.getParam("advection.u")
    v = runparams.getParam("advection.v")
    
    # the timestep is min(dx/|u|, dy|v|)
    xtmp = myData.grid.dx/max(abs(u),SMALL)
    ytmp = myData.grid.dy/max(abs(v),SMALL)

    dt = cfl*min(xtmp, ytmp)

    return dt



