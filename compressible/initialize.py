import numpy
import mesh.patch as patch
from util import runparams

def initialize():
    """ 
    initialize the grid and variables for compressible flow
    """
    import vars

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

    # first figure out the boundary conditions.  Note: the action
    # can depend on the variable (for reflecting BCs)
    xlb_type = runparams.getParam("mesh.xlboundary")
    xrb_type = runparams.getParam("mesh.xrboundary")
    ylb_type = runparams.getParam("mesh.ylboundary")
    yrb_type = runparams.getParam("mesh.yrboundary")    
    
    bcObj = patch.bcObject(xlb=xlb_type, xrb=xrb_type,
                           ylb=ylb_type, yrb=yrb_type)

    # density and energy
    myData.registerVar("density", bcObj)
    myData.registerVar("energy", bcObj)

    # for velocity, if we are reflecting, we need odd reflection
    # in the normal direction.

    # x-momentum -- if we are reflecting in x, then we need to
    # reflect odd
    bcObj_xodd = patch.bcObject(xlb=xlb_type, xrb=xrb_type,
                                ylb=ylb_type, yrb=yrb_type,
                                oddReflectDir="x")

    myData.registerVar("x-momentum", bcObj_xodd)    


    # y-momentum -- if we are reflecting in y, then we need to
    # reflect odd
    bcObj_yodd = patch.bcObject(xlb=xlb_type, xrb=xrb_type,
                                ylb=ylb_type, yrb=yrb_type,
                                oddReflectDir="y")

    myData.registerVar("y-momentum", bcObj_yodd)    


    # store the EOS gamma as an auxillary quantity so we can have a
    # self-contained object stored in output files to make plots
    gamma = runparams.getParam("eos.gamma")
    myData.setAux("gamma", gamma)
        
    myData.create()


    vars.idens = myData.vars.index("density")
    vars.ixmom = myData.vars.index("x-momentum")
    vars.iymom = myData.vars.index("y-momentum")
    vars.iener = myData.vars.index("energy")

    print myData

    return myGrid, myData
