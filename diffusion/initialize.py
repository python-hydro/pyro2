import numpy

import mesh.patch as patch
from util import msg, runparams

def initialize():
    """ 
    initialize the grid and variables for diffusion
    """

    # setup the grid
    nx = runparams.getParam("mesh.nx")
    ny = runparams.getParam("mesh.ny")

    xmin = runparams.getParam("mesh.xmin")
    xmax = runparams.getParam("mesh.xmax")
    ymin = runparams.getParam("mesh.ymin")
    ymax = runparams.getParam("mesh.ymax")
    
    myGrid = patch.Grid2d(nx, ny, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, ng=1)


    # create the variables

    # first figure out the boundary conditions -- we allow periodic,
    # Dirichlet, and Neumann.

    xlb_type = runparams.getParam("mesh.xlboundary")
    xrb_type = runparams.getParam("mesh.xrboundary")
    ylb_type = runparams.getParam("mesh.ylboundary")
    yrb_type = runparams.getParam("mesh.yrboundary")

    bcparam = []
    for bc in [xlb_type, xrb_type, ylb_type, yrb_type]:
        if   (bc == "periodic"): bcparam.append("periodic")
        elif (bc == "neumann"):  bcparam.append("neumann")
        elif (bc == "dirichlet"):  bcparam.append("dirichlet")
        else:
            msg.fail("invalid BC")


    bcObj = patch.BCObject(xlb=bcparam[0], xrb=bcparam[1], 
                           ylb=bcparam[2], yrb=bcparam[3])    


    myData = patch.CellCenterData2d(myGrid)

    myData.registerVar("phi", bcObj)

    myData.create()

    return myGrid, myData

                                                                                    
