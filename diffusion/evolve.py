from util import runparams
import multigrid

def evolve(myd, dt):
    """ diffusion through dt using C-N implicit solve with multigrid """

    
    phi = myData.getVarPtr("phi")
    myg = myData.grid

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
    f[mg.imin:mg.imax+1,mg.jmin:mg.jmax+1] = \
        phi[myg.imin:myg.imax+1,myg.jmin:myg.jmax+1] + 0.5*dt*k * \
        ((phi[myg.imin+1:myg.imax+2,myg.jmin:myg.jmax+1] +
          phi[myg.imin-1:myg.imax  ,myg.jmin:myg.jmax+1] -
          2.0*phi[myg.imin:myg.imax+1,myg.jmin:myg.jmax+1])/myg.dx**2 +
         (phi[myg.imin:myg.imax+1,myg.jmin+1:myg.jmax+2] +
          phi[myg.imin:myg.imax+1,myg.jmin-1:myg.jmax  ] -
          2.0*phi[myg.imin:myg.imax+1,myg.jmin:myg.jmax+1])/myg.dy**2)

    mg.initRHS(f)


    # initial guess is the current solution
    mg.initSolution(phi)


    # solve the MG problem for the updated phi
    mg.solve(rtol=1.e-10)


    # update the solution
    phi[:,:] = mg.getSolution()
