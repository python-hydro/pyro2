import numpy

import evolve
import mesh.patch as patch
import multigrid.multigrid as multigrid
import timestep
from util import runparams

def preevolve(myData):
    """ 
    preevolve is called before we being the timestepping loop.  For
    the incompressible solver, this does an initial projection on the
    velocity field and then goes through the full evolution to get the
    value of phi.  The fluid state (u, v) is then reset to values
    before this evolve.
    """

    myg = myData.grid

    u = myData.getVarPtr("x-velocity")
    v = myData.getVarPtr("y-velocity")

    myData.fillBC("x-velocity")
    myData.fillBC("y-velocity")


    # 1. do the initial projection.  This makes sure that our original
    # velocity field satisties div U = 0

    # next create the multigrid object.  We want Neumann BCs on phi
    # at solid walls and periodic on phi for periodic BCs
    MG = multigrid.ccMG2d(myg.nx, myg.ny,
                          xlBCtype="periodic", xrBCtype="periodic",
                          ylBCtype="periodic", yrBCtype="periodic",
                          xmin=myg.xmin, xmax=myg.xmax,
                          ymin=myg.ymin, ymax=myg.ymax,
                          verbose=0)

    # first compute divU
    divU = MG.solnGrid.scratchArray()

    divU[MG.ilo:MG.ihi+1,MG.jlo:MG.jhi+1] = \
        0.5*(u[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] - 
             u[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1])/myg.dx + \
        0.5*(v[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] - 
             v[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])/myg.dy


    # solve L phi = DU

    # initialize our guess to the solution
    MG.initZeros()

    # setup the RHS of our Poisson equation
    MG.initRHS(divU)

    # solve
    MG.solve(rtol=1.e-10)

    # store the solution in our myData object -- include a single
    # ghostcell
    phi = myData.getVarPtr("phi")
    solution = MG.getSolution()

    phi[myg.ilo-1:myg.ihi+2,myg.jlo-1:myg.jhi+2] = \
        solution[MG.ilo-1:MG.ihi+2,MG.jlo-1:MG.jhi+2]

    # compute the cell-centered gradient of phi and update the 
    # velocities
    gradp_x = myg.scratchArray()
    gradp_y = myg.scratchArray()

    gradp_x[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
        0.5*(phi[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
             phi[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1])/myg.dx

    gradp_y[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
        0.5*(phi[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
             phi[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])/myg.dy

    u[:,:] -= gradp_x
    v[:,:] -= gradp_y

    # fill the ghostcells
    myData.fillBC("x-velocity")
    myData.fillBC("y-velocity")


    # 2. now get an approximation to gradp at n-1/2 by going through the
    # evolution.

    # store the current solution
    copyData = patch.ccDataClone(myData)

    # get the timestep
    dt = timestep.timestep(copyData)

    # evolve
    evolve.evolve(copyData, dt)

    # update gradp_x and gradp_y in our main data object
    gp_x = myData.getVarPtr("gradp_x")
    gp_y = myData.getVarPtr("gradp_y")

    new_gp_x = copyData.getVarPtr("gradp_x")
    new_gp_y = copyData.getVarPtr("gradp_y")

    gp_x[:,:] = new_gp_x[:,:]
    gp_y[:,:] = new_gp_y[:,:]


    print "done with the pre-evolution"
