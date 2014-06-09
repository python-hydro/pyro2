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
    
    my_grid = patch.Grid2d(nx, ny, 
                          xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, 
                          ng=4)


    # create the variables
    my_data = patch.CellCenterData2d(my_grid, runtime_parameters=rp)


    # define solver specific boundary condition routines
    patch.define_bc("hse", BC.user)


    # first figure out the boundary conditions.  Note: the action
    # can depend on the variable (for reflecting BCs)
    xlb_type = rp.get_param("mesh.xlboundary")
    xrb_type = rp.get_param("mesh.xrboundary")
    ylb_type = rp.get_param("mesh.ylboundary")
    yrb_type = rp.get_param("mesh.yrboundary")    
    
    bc = patch.BCObject(xlb=xlb_type, xrb=xrb_type,
                        ylb=ylb_type, yrb=yrb_type)

    # density and energy
    my_data.register_var("density", bc)
    my_data.register_var("energy", bc)

    # for velocity, if we are reflecting, we need odd reflection
    # in the normal direction.

    # x-momentum -- if we are reflecting in x, then we need to
    # reflect odd
    bc_xodd = patch.BCObject(xlb=xlb_type, xrb=xrb_type,
                             ylb=ylb_type, yrb=yrb_type,
                             odd_reflect_dir="x")

    my_data.register_var("x-momentum", bc_xodd)    


    # y-momentum -- if we are reflecting in y, then we need to
    # reflect odd
    bc_yodd = patch.BCObject(xlb=xlb_type, xrb=xrb_type,
                             ylb=ylb_type, yrb=yrb_type,
                             odd_reflect_dir="y")

    my_data.register_var("y-momentum", bc_yodd)    


    # store the EOS gamma as an auxillary quantity so we can have a
    # self-contained object stored in output files to make plots
    gamma = rp.get_param("eos.gamma")
    my_data.set_aux("gamma", gamma)

    # initialize the EOS gamma
    eos.init(gamma)
        
    my_data.create()


    vars.idens = my_data.vars.index("density")
    vars.ixmom = my_data.vars.index("x-momentum")
    vars.iymom = my_data.vars.index("y-momentum")
    vars.iener = my_data.vars.index("energy")

    print my_data

    return my_grid, my_data
