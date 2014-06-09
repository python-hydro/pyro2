import numpy

import evolve
import mesh.patch as patch
import multigrid.multigrid as multigrid
import timestep

def preevolve(my_data):
    """ 
    preevolve is called before we being the timestepping loop.  For
    the incompressible solver, this does an initial projection on the
    velocity field and then goes through the full evolution to get the
    value of phi.  The fluid state (u, v) is then reset to values
    before this evolve.
    """

    myg = my_data.grid

    u = my_data.get_var("x-velocity")
    v = my_data.get_var("y-velocity")

    my_data.fill_BC("x-velocity")
    my_data.fill_BC("y-velocity")


    # 1. do the initial projection.  This makes sure that our original
    # velocity field satisties div U = 0

    # next create the multigrid object.  We want Neumann BCs on phi
    # at solid walls and periodic on phi for periodic BCs
    MG = multigrid.CellCenterMG2d(myg.nx, myg.ny,
                                  xl_BC_type="periodic", xr_BC_type="periodic",
                                  yl_BC_type="periodic", yr_BC_type="periodic",
                                  xmin=myg.xmin, xmax=myg.xmax,
                                  ymin=myg.ymin, ymax=myg.ymax,
                                  verbose=0)

    # first compute divU
    divU = MG.soln_grid.scratch_array()

    divU[MG.ilo:MG.ihi+1,MG.jlo:MG.jhi+1] = \
        0.5*(u[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] - 
             u[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1])/myg.dx + \
        0.5*(v[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] - 
             v[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])/myg.dy


    # solve L phi = DU

    # initialize our guess to the solution
    MG.init_zeros()

    # setup the RHS of our Poisson equation
    MG.init_RHS(divU)

    # solve
    MG.solve(rtol=1.e-10)

    # store the solution in our my_data object -- include a single
    # ghostcell
    phi = my_data.get_var("phi")
    solution = MG.get_solution()

    phi[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2] = \
        solution[MG.ilo-1:MG.ihi+2,MG.jlo-1:MG.jhi+2]

    # compute the cell-centered gradient of phi and update the 
    # velocities
    gradp_x = myg.scratch_array()
    gradp_y = myg.scratch_array()

    gradp_x[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
        0.5*(phi[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
             phi[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1])/myg.dx

    gradp_y[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
        0.5*(phi[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
             phi[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])/myg.dy

    u[:,:] -= gradp_x
    v[:,:] -= gradp_y

    # fill the ghostcells
    my_data.fill_BC("x-velocity")
    my_data.fill_BC("y-velocity")


    # 2. now get an approximation to gradp at n-1/2 by going through the
    # evolution.

    # store the current solution
    copyData = patch.cell_center_data_clone(my_data)

    # get the timestep
    dt = timestep.timestep(copyData)

    # evolve
    evolve.evolve(copyData, dt)

    # update gradp_x and gradp_y in our main data object
    gp_x = my_data.get_var("gradp_x")
    gp_y = my_data.get_var("gradp_y")

    new_gp_x = copyData.get_var("gradp_x")
    new_gp_y = copyData.get_var("gradp_y")

    gp_x[:,:] = new_gp_x[:,:]
    gp_y[:,:] = new_gp_y[:,:]


    print "done with the pre-evolution"
