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
    
    myGrid = patch.Grid2d(nx, ny, 
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

    my_data = patch.CellCenterData2d(myGrid, runtime_parameters=rp)

    # velocities
    my_data.registerVar("x-velocity", bcObj_xodd)
    my_data.registerVar("y-velocity", bcObj_yodd)

    # phi -- used for the projections
    my_data.registerVar("phi-MAC", bcObj)
    my_data.registerVar("phi", bcObj)
    my_data.registerVar("gradp_x", bcObj)
    my_data.registerVar("gradp_y", bcObj)

    my_data.create()

    return myGrid, my_data
