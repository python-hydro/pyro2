import numpy
import mesh.patch as patch
from util import runparams



def initialize():
    """ 
    initialize the grid and variables for compressible flow
    """

    # setup the grid
    nx = runparams.getParam("mesh.nx")
    ny = runparams.getParam("mesh.ny")

    xmin = runparams.getParam("mesh.xmin")
    xmax = runparams.getParam("mesh.xmax")
    ymin = runparams.getParam("mesh.ymin")
    ymax = runparams.getParam("mesh.ymax")
    
    myGrid = patch.grid2d(nx, ny, 
                          xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, 
                          ng=4)


    # create the variables
    myData = patch.ccData2d(myGrid)

    # first figure out the boundary conditions -- we need to translate
    # between the descriptive type of the boundary specified by the
    # user and the action that will be performed by the fillBC routine.
    # The actions can vary depending on the variable (especially for
    # velocity and reflecting BCs).

    xlb_type = runparams.getParam("mesh.xlboundary")
    xrb_type = runparams.getParam("mesh.xrboundary")
    ylb_type = runparams.getParam("mesh.ylboundary")
    yrb_type = runparams.getParam("mesh.yrboundary")    
    
    # density and energy
    bcparam = []
    for bc in [xlb_type, xrb_type, ylb_type, yrb_type]:
        if   (bc == "periodic"): bcparam.append("periodic")
        elif (bc == "reflect"):  bcparam.append("reflect-even")
        elif (bc == "outflow"):  bcparam.append("outflow")

    bcObj = patch.bcObject(xlb=bcparam[0], xrb=bcparam[1], 
                           ylb=bcparam[2], yrb=bcparam[3])    

    myData.registerVar("density", bcObj)
    myData.registerVar("energy", bcObj)

    # for velocity, if we are reflecting, we need odd reflection
    # in the normal direction.

    # x-momentum -- if we are reflecting in x, then we need to
    # reflect odd
    bcparam_x = bcparam.copy()
    if (xlb_type == "reflect"): bcparam_x[0] = "reflect-odd"
    if (xrb_type == "reflect"): bcparam_x[1] = "reflect-odd"

    bcObj = patch.bcObject(xlb=bcparam_x[0], xrb=bcparam_x[1], 
                           ylb=bcparam_x[2], yrb=bcparam_x[3])    

    myData.registerVar("x-momentum", bcObj)    


    # y-momentum -- if we are reflecting in y, then we need to
    # reflect odd
    bcparam_y = bcparam.copy()
    if (ylb_type == "reflect"): bcparam_y[2] = "reflect-odd"
    if (yrb_type == "reflect"): bcparam_y[3] = "reflect-odd"

    bcObj = patch.bcObject(xlb=bcparam_y[0], xrb=bcparam_y[1], 
                           ylb=bcparam_y[2], yrb=bcparam_y[3])    

    myData.registerVar("y-momentum", bcObj)    

        
    myData.create()

    return myGrid, myData
