from util import runparams
import multigrid

def evolve(myd, dt):
    """ diffusion through dt using C-N implicit solve with multigrid """

    
    phi = myData.getVarPtr("phi")

    # diffusion coefficient
    k = runparams.getparam("diffusion.k")
    

    # setup the MG object
    mg multigrid.ccMG2d(myd,nx, myd.ny,
                        xmin=myd.xmin, xmax=myd.xmax, 
                        ymin=myd.ymin, ymax=myd.ymax,
                        xlBCtype=myd.BCs['phi'].xlb, 
                        xrBCtype=myd.BCs['phi'].xrb, 
                        ylBCtype=myd.BCs['phi'].ylb, 
                        yrBCtype=myd.BCs['phi'].yrb, 
                        alpha=1.0, beta=-0.5*dt*k)


    # form the RHS: f = phi + (dt/2) k L phi  (where L is the Laplacian)
    f = mg.solnGrid.scratchArray()
                        
    mg.initRHS(f)


    # initial guess is the current solution
    mg.initSolution(phi)


    # solve the MG problem for the updated phi
    mg.solve(rtol=1.e-10)


    # update the solution
    phi[:,:] = mg.getSolution()
