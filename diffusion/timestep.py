from util import runparams

"""
The diffusion timestep module computes the timestep using the explicit
timestep constraint as the starting point.  We then multiply by the
CFL number to get the timestep.  Since we are doing an implicit
discretization, we do not require CFL < 1.
"""

def timestep(my_data):
    """ 
    compute the CFL timestep for the current patch.
    """

    cfl = runparams.getParam("driver.cfl")
    
    k = runparams.getParam("diffusion.k")
    
    # the timestep is min(dx**2/k, dy**2/k)
    xtmp = my_data.grid.dx**2/k
    ytmp = my_data.grid.dy**2/k

    dt = cfl*min(xtmp, ytmp)

    return dt



