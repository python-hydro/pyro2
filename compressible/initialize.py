import numpy

import BC
import eos
import mesh.patch as patch

def initialize(rp):
    """ 
    initialize the grid and variables for compressible flow
    """
    import vars

    # setup the grid
    nx = rp.get_param("mesh.nx")
    ny = rp.get_param("mesh.ny")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")
    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")
    
    myGrid = patch.Grid2d(nx, ny, 
                          xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, 
                          ng=4)


    # create the variables
    my_data = patch.CellCenterData2d(myGrid, runtime_parameters=rp)


    # define solver specific boundary condition routines
    patch.defineBC("hse", BC.user)


    # first figure out the boundary conditions.  Note: the action
    # can depend on the variable (for reflecting BCs)
    xlb_type = rp.get_param("mesh.xlboundary")
    xrb_type = rp.get_param("mesh.xrboundary")
    ylb_type = rp.get_param("mesh.ylboundary")
    yrb_type = rp.get_param("mesh.yrboundary")    
    
    bcObj = patch.BCObject(xlb=xlb_type, xrb=xrb_type,
                           ylb=ylb_type, yrb=yrb_type)

    # density and energy
    my_data.registerVar("density", bcObj)
    my_data.registerVar("energy", bcObj)

    # for velocity, if we are reflecting, we need odd reflection
    # in the normal direction.

    # x-momentum -- if we are reflecting in x, then we need to
    # reflect odd
    bcObj_xodd = patch.BCObject(xlb=xlb_type, xrb=xrb_type,
                                ylb=ylb_type, yrb=yrb_type,
                                oddReflectDir="x")

    my_data.registerVar("x-momentum", bcObj_xodd)    


    # y-momentum -- if we are reflecting in y, then we need to
    # reflect odd
    bcObj_yodd = patch.BCObject(xlb=xlb_type, xrb=xrb_type,
                                ylb=ylb_type, yrb=yrb_type,
                                oddReflectDir="y")

    my_data.registerVar("y-momentum", bcObj_yodd)    


    # store the EOS gamma as an auxillary quantity so we can have a
    # self-contained object stored in output files to make plots
    gamma = rp.get_param("eos.gamma")
    my_data.setAux("gamma", gamma)

    # initialize the EOS gamma
    eos.init(gamma)
        
    my_data.create()


    vars.idens = my_data.vars.index("density")
    vars.ixmom = my_data.vars.index("x-momentum")
    vars.iymom = my_data.vars.index("y-momentum")
    vars.iener = my_data.vars.index("energy")

    print my_data

    return myGrid, my_data
