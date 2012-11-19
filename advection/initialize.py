import numpy
import mesh.patch as patch
from util import runparams



def initialize():
    """ 
    initialize the grid and variables for advection 
    """

    # setup the grid
    nx = runparams.getParam("mesh.nx")
    ny = runparams.getParam("mesh.ny")

    xmin = runparams.getParam("mesh.xmin")
    xmax = runparams.getParam("mesh.xmax")
    ymin = runparams.getParam("mesh.ymin")
    ymax = runparams.getParam("mesh.ymax")
    
    myGrid = patch.grid2d(nx, ny, 
                          xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, ng=4)


    # create the variables

    # first figure out the boundary conditions -- we need to translate
    # between the descriptive type of the boundary specified by the
    # user and the action that will be performed by the fillBC routine.
    # Usually the actions can vary depending on the variable, but we
    # only have one variable.
    xlb_type = runparams.getParam("mesh.xlboundary")
    xrb_type = runparams.getParam("mesh.xrboundary")
    ylb_type = runparams.getParam("mesh.ylboundary")
    yrb_type = runparams.getParam("mesh.yrboundary")

    bcObj = patch.bcObject(xlb=xlb_type, xrb=xrb_type, 
                           ylb=ylb_type, yrb=yrb_type)

    myData = patch.ccData2d(myGrid)

    myData.registerVar("density", bcObj)

    myData.create()

    return myGrid, myData
