import numpy

import mesh.patch as patch
from util import msg, runparams

def initialize(rp):
    """ 
    initialize the grid and variables for diffusion
    """

    # setup the grid
    nx = rp.get_param("mesh.nx")
    ny = rp.get_param("mesh.ny")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")
    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")
    
    myGrid = patch.Grid2d(nx, ny, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, ng=1)


    # create the variables

    # first figure out the boundary conditions -- we allow periodic,
    # Dirichlet, and Neumann.

    xlb_type = rp.get_param("mesh.xlboundary")
    xrb_type = rp.get_param("mesh.xrboundary")
    ylb_type = rp.get_param("mesh.ylboundary")
    yrb_type = rp.get_param("mesh.yrboundary")

    bcparam = []
    for bc in [xlb_type, xrb_type, ylb_type, yrb_type]:
        if   (bc == "periodic"): bcparam.append("periodic")
        elif (bc == "neumann"):  bcparam.append("neumann")
        elif (bc == "dirichlet"):  bcparam.append("dirichlet")
        else:
            msg.fail("invalid BC")


    bcObj = patch.BCObject(xlb=bcparam[0], xrb=bcparam[1], 
                           ylb=bcparam[2], yrb=bcparam[3])    


    my_data = patch.CellCenterData2d(myGrid, runtime_parameters=rp)

    my_data.registerVar("phi", bcObj)

    my_data.create()

    return myGrid, my_data
