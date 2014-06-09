import numpy

import multigrid.multigrid as multigrid

def evolve(myd, dt):
    """ diffusion through dt using C-N implicit solve with multigrid """

    myd.fill_BC_all()
    phi = myd.get_var("phi")
    myg = myd.grid

    # diffusion coefficient
    k = myd.rp.get_param("diffusion.k")
    

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
    mg = multigrid.CellCenterMG2d(myg.nx, myg.ny,
                                  xmin=myg.xmin, xmax=myg.xmax, 
                                  ymin=myg.ymin, ymax=myg.ymax,
                                  xlBCtype=myd.BCs['phi'].xlb, 
                                  xrBCtype=myd.BCs['phi'].xrb, 
                                  ylBCtype=myd.BCs['phi'].ylb, 
                                  yrBCtype=myd.BCs['phi'].yrb, 
                                  alpha=1.0, beta=0.5*dt*k, 
                                  verbose=0)


    # form the RHS: f = phi + (dt/2) k L phi  (where L is the Laplacian)
    f = mg.soln_grid.scratch_array()
    f[mg.ilo:mg.ihi+1,mg.jlo:mg.jhi+1] = \
        phi[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] + 0.5*dt*k * \
        ((phi[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] +
          phi[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1] -
          2.0*phi[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dx**2 +
         (phi[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] +
          phi[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ] -
          2.0*phi[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/myg.dy**2)


    mg.init_RHS(f)


    # initial guess is the current solution
    mg.init_zeros()


    # solve the MG problem for the updated phi
    mg.solve(rtol=1.e-10)
    #mg.smooth(mg.nlevels-1,100)


    # update the solution
    phi[:,:] = mg.get_solution()
