import numpy
import mesh.patch as patch
from util import runparams
import multigrid.multigrid as multigrid

def preevolve(myData, dt):
    """ 
    preevolve is called before we being the timestepping loop.  For the
    incompressible solver, this does the initial projection to get the
    value of phi
    """

    nproj = runparams.getParam("incompressible.num_init_proj")

    myg = myData.grid

    u = myData.getVarPtr("x-velocity")
    v = myData.getVarPtr("y-velocity")

    # do the initial projection.  

    # next create the multigrid object.  We want Neumann BCs on phi
    # at solid walls and periodic on phi for periodic BCs
    MG = multigrid.ccMg2d(myg.nx, myg.ny,
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
    iproj = 0
    while (iproj < nproj):

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
            solution[MG.ilo-1:MG.ihi+2,MG.jlo-1:MG.jlo+2]

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
        
        # recompute the divergence of the velocity for the next
        # iteration
        divU[MG.ilo:MG.ihi+1,MG.jlo:MG.jhi+1] = \
            0.5*(u[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] - 
                 u[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1])/myg.dx + \
            0.5*(v[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] - 
                 v[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ])/myg.dy

        iproj += 1
        
