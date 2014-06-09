SMALL = 1.e-12

def timestep(my_data):
    """
    Computes the advective timestep (CFL) constraint.  We use the 
    driver.cfl parameter to control what fraction of the CFL
    step we actually take.
    """
    
    rp = my_data.rp

    cfl = rp.get_param("driver.cfl")
    
    u = rp.get_param("advection.u")
    v = rp.get_param("advection.v")
    
    # the timestep is min(dx/|u|, dy|v|)
    xtmp = my_data.grid.dx/max(abs(u),SMALL)
    ytmp = my_data.grid.dy/max(abs(v),SMALL)

    dt = cfl*min(xtmp, ytmp)

    return dt



