import numpy

import mesh.patch as patch
from util import runparams

def initialize():
    """ 
    initialize the grid and variables for incompressible flow
    """

    # setup the grid
    nx = runparams.getParam("mesh.nx")
    ny = runparams.getParam("mesh.ny")

    xmin = runparams.getParam("mesh.xmin")
    xmax = runparams.getParam("mesh.xmax")
    ymin = runparams.getParam("mesh.ymin")
    ymax = runparams.getParam("mesh.ymax")
    
    myGrid = patch.Grid2d(nx, ny, 
                          xmin=xmin, xmax=xmax, 
                          ymin=ymin, ymax=ymax, ng=4)


    # create the variables

    # first figure out the BCs
    xlb_type = runparams.getParam("mesh.xlboundary")
    xrb_type = runparams.getParam("mesh.xrboundary")
    ylb_type = runparams.getParam("mesh.ylboundary")
    yrb_type = runparams.getParam("mesh.yrboundary")

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

    my_data = patch.CellCenterData2d(myGrid)

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
