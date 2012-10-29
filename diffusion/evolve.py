from util import runparams
import multigrid.multigrid as multigrid
import numpy

def evolve(myd, dt):
    """ diffusion through dt using C-N implicit solve with multigrid """

    myd.fillBCAll()
    phi = myd.getVarPtr("phi")
    myg = myd.grid

    # diffusion coefficient
    k = runparams.getParam("diffusion.k")
    

    # setup the MG object -- we want to solve a Helmholtz equation
    # equation of the form:
    # (alpha - beta L) phi = f
    #
    # with alpha = 1
    #      beta  = (dt/2) k
    #      f     = phi + (dt/2) k L phi
    #
    # this is the form that arises with a Crank-Nicolson discretization
    # of the diffusion equation.
    mg = multigrid.ccMG2d(myg.nx, myg.ny,
                          xmin=myg.xmin, xmax=myg.xmax, 
                          ymin=myg.ymin, ymax=myg.ymax,
                          xlBCtype=myd.BCs['phi'].xlb, 
                          xrBCtype=myd.BCs['phi'].xrb, 
                          ylBCtype=myd.BCs['phi'].ylb, 
                          yrBCtype=myd.BCs['phi'].yrb, 
                          alpha=1.0, beta=0.5*dt*k, 
                          verbose=1)


    # form the RHS: f = phi + (dt/2) k L phi  (where L is the Laplacian)
    f = mg.solnGrid.scratchArray()
    f[mg.ilo:mg.ihi+1,mg.jlo:mg.jhi+1] = \
        phi[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] + 0.5*dt*k * \
        ((phi[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] +
          phi[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1] -
          2.0*phi[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx**2 +
         (phi[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] +
          phi[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ] -
          2.0*phi[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dy**2)


    mg.initRHS(f)


    # initial guess is the current solution
    mg.initZeros()


    # solve the MG problem for the updated phi
    mg.solve(rtol=1.e-10)
    #mg.smooth(mg.nlevels-1,100)


    # update the solution
    phi[:,:] = mg.getSolution()
