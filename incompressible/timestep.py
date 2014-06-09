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
    
    u = my_data.getVarPtr("x-velocity")
    v = my_data.getVarPtr("y-velocity")
    
    # the timestep is min(dx/|u|, dy|v|)
    xtmp = my_data.grid.dx/(abs(u))
    ytmp = my_data.grid.dy/(abs(v))

    dt = cfl*min(xtmp.min(), ytmp.min())

    return dt



