import numpy

import mesh.patch as patch

def initialize(rp):
    """ 
    initialize the grid and variables for advection 
    """

    # setup the grid
    nx = rp.get_param("mesh.nx")
    ny = rp.get_param("mesh.ny")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")
    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")
    
    my_grid = patch.Grid2d(nx, ny, 
                          xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, ng=4)


    # create the variables

    # first figure out the boundary conditions -- we need to translate
    # between the descriptive type of the boundary specified by the
    # user and the action that will be performed by the fillBC routine.
    # Usually the actions can vary depending on the variable, but we
    # only have one variable.
    xlb_type = rp.get_param("mesh.xlboundary")
    xrb_type = rp.get_param("mesh.xrboundary")
    ylb_type = rp.get_param("mesh.ylboundary")
    yrb_type = rp.get_param("mesh.yrboundary")

    bcObj = patch.BCObject(xlb=xlb_type, xrb=xrb_type, 
                           ylb=ylb_type, yrb=yrb_type)

    my_data = patch.CellCenterData2d(my_grid, runtime_parameters=rp)

    my_data.registerVar("density", bcObj)

    my_data.create()

    return my_grid, my_data
