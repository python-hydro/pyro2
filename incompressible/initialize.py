import numpy

import mesh.patch as patch

def initialize(rp):
    """ 
    initialize the grid and variables for incompressible flow
    """

    # setup the grid
    nx = rp.get_param("mesh.nx")
    ny = rp.get_param("mesh.ny")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")
    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")
    
    my_grid = patch.Grid2d(nx, ny, 
                          xmin=xmin, xmax=xmax, 
                          ymin=ymin, ymax=ymax, ng=4)


    # create the variables

    # first figure out the BCs
    xlb_type = rp.get_param("mesh.xlboundary")
    xrb_type = rp.get_param("mesh.xrboundary")
    ylb_type = rp.get_param("mesh.ylboundary")
    yrb_type = rp.get_param("mesh.yrboundary")

    bcObj = patch.BCObject(xlb=xlb_type, xrb=xrb_type, 
                           ylb=ylb_type, yrb=yrb_type)

    # if we are reflecting, we need odd reflection in the normal
    # directions for the velocity
    bcObj_xodd = patch.BCObject(xlb=xlb_type, xrb=xrb_type, 
                                ylb=ylb_type, yrb=yrb_type,
                                oddReflectDir="x")

    bcObj_yodd = patch.BCObject(xlb=xlb_type, xrb=xrb_type, 
                                ylb=ylb_type, yrb=yrb_type,
                                oddReflectDir="y")

    my_data = patch.CellCenterData2d(my_grid, runtime_parameters=rp)

    # velocities
    my_data.register_var("x-velocity", bcObj_xodd)
    my_data.register_var("y-velocity", bcObj_yodd)

    # phi -- used for the projections
    my_data.register_var("phi-MAC", bcObj)
    my_data.register_var("phi", bcObj)
    my_data.register_var("gradp_x", bcObj)
    my_data.register_var("gradp_y", bcObj)

    my_data.create()

    return my_grid, my_data
