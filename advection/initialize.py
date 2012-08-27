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
    
    myGrid = patch.grid2d(nx, ny, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)


    # create the variables

    # first figure out the boundary conditions -- we need to translate
    # between the descriptive type of the boundary specified by the
    # user and the action that will be performed by the fillBC routine.
    # Usually the actions can vary depending on the variable, but we
    # only have one variable.

    xlb_type = runparams.getParam("mesh.xlboundary")
    if   (xlb_type == "periodic"): xlb = "periodic"
    elif (xlb_type == "reflect"):  xlb = "reflect-even"
    elif (xlb_type == "outflow"):  xlb = "outflow"

    xrb_type = runparams.getParam("mesh.xrboundary")
    if   (xrb_type == "periodic"): xrb = "periodic"
    elif (xrb_type == "reflect"):  xrb = "reflect-even"
    elif (xrb_type == "outflow"):  xrb = "outflow"

    ylb_type = runparams.getParam("mesh.ylboundary")
    if   (ylb_type == "periodic"): ylb = "periodic"
    elif (ylb_type == "reflect"):  ylb = "reflect-even"
    elif (ylb_type == "outflow"):  ylb = "outflow"

    yrb_type = runparams.getParam("mesh.yrboundary")
    if   (yrb_type == "periodic"): yrb = "periodic"
    elif (yrb_type == "reflect"):  yrb = "reflect-even"
    elif (yrb_type == "outflow"):  yrb = "outflow"

    bcObj = patch.bcObject(xlb=xlb, xrb=xrb, ylb=ylb, yrb=yrb)

    myData = patch.ccData2d(myGrid)

    myData.registerVar("density", bcObj)

    myData.create()

    return myGrid, myData

                                                                                    
