"""

The multigrid module provides a framework for solving elliptic
problems.  A multigrid object is just a list of grids, from the finest
mesh down (by factors of two) to a single interior zone (each grid has
the same number of guardcells).

The main multigrid class (MGcc) is setup to solve a constant-coefficient
Helmholtz equation:

(alpha - beta L) phi = f

where L is the Laplacian and alpha and beta are constants.  If alpha =
0 and beta = -1, then this is the Poisson equation.

We support homogeneous Dirichlet or Neumann BCs, or on periodic domain.

The general usage is as follows:

> a = multigrid.CellCenterMG2d(nx, ny, verbose=1, alpha=alpha, beta=beta)

this creates the multigrid object a, with a finest grid of nx by ny
zones and the default boundary condition types.  alpha and beta are
the coefficients of the Helmholtz equation.  Setting verbose = 1
causing debugging information to be output, so you can see the
residual errors in each of the V-cycles.

> a.init_solution(zeros((nx, ny), numpy.float64))

this initializes the solution vector with zeros

> a.init_RHS(zeros((nx, ny), numpy.float64))

this initializes the RHS on the finest grid to 0 (Laplace's equation).
Any RHS can be set by passing through an array of nx values here.

Then to solve, you just do:

> a.solve(rtol = 1.e-10)

where rtol is the desired tolerance (relative difference in solution from
one cycle to the next).

to access the final solution, use the getSolution method

v = a.get_solution()

For convenience, the grid information on the solution level is available as
attributes to the class,

a.ilo, a.ihi, a.jlo, a.jhi are the indices bounding the interior
of the solution array (i.e. excluding the guardcells).

a.x and a.y are the coordinate arrays
a.dx and a.dy are the grid spacings

"""

import math

import numpy
import pylab
import matplotlib

import mesh.patch as patch

def error(myg, r):

    # L2 norm of elements in r, multiplied by dx*dy to
    # normalize
    return numpy.sqrt(myg.dx*myg.dy*
                      numpy.sum((r[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]**2).flat))


class CellCenterMG2d:
    """ 
    The main multigrid class for cell-centered data.

    We require that nx = ny be a power of 2 and dx = dy, for
    simplicity
    """
    
    def __init__(self, nx, ny, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
                 xlBCtype="dirichlet", xrBCtype="dirichlet",
                 ylBCtype="dirichlet", yrBCtype="dirichlet",
                 alpha=0.0, beta=-1.0,
                 nsmooth=10, nsmooth_bottom=50,
                 verbose=0, 
                 trueFunc=None, vis=0, vis_title=""):

        if nx != ny:
            print "ERROR: multigrid currently requires nx = ny"
            return -1
        
        self.nx = nx
        self.ny = ny        
        
        self.ng = 1

        self.xmin = xmin
        self.xmax = xmax

        self.ymin = ymin
        self.ymax = ymax        

        if (xmax-xmin) != (ymax-ymin):
            print "ERROR: multigrid currently requires a square domain"
            return -1
        
        self.alpha = alpha
        self.beta = beta

        self.nsmooth = nsmooth
        self.nbottomSmooth = nsmooth_bottom

        self.maxCycles = 100
        
        self.verbose = verbose

        # for visualization purposes, we can set a function name that
        # provides the true solution to our elliptic problem.
        if not trueFunc == None:
            self.trueFunc = trueFunc

        # a small number used in computing the error, so we don't divide by 0
        self.small = 1.e-16
        
        # keep track of whether we've initialized the solution
        self.initializedSolution = 0
        self.initializedRHS = 0
        
        # assume that self.nx = 2^(nlevels-1) and that nx = ny
        # this defines nlevels such that we end exactly on a 2x2 grid
        self.nlevels = int(math.log(self.nx)/math.log(2.0)) 

        # a multigrid object will be a list of grids
        self.grids = []

        # create the grids.  Here, self.grids[0] will be the coarsest
        # grid and self.grids[nlevel-1] will be the finest grid
        # we store the solution, v, the rhs, f.
        i = 0
        nx_t = ny_t = 2

        if self.verbose:
            print "alpha = ", self.alpha
            print "beta  = ", self.beta

        while (i < self.nlevels):
            
            # create the grid
            myGrid = patch.Grid2d(nx_t, ny_t, ng=self.ng,
                                  xmin=xmin, xmax=xmax,
                                  ymin=ymin, ymax=ymax)

            # add a CellCenterData2d object for this level to our list
            self.grids.append(patch.CellCenterData2d(myGrid, dtype=numpy.float64))

            # create the boundary condition object
            bcObj = patch.BCObject(xlb=xlBCtype, xrb=xrBCtype,
                                   ylb=ylBCtype, yrb=yrBCtype)

            self.grids[i].register_var("v", bcObj)
            self.grids[i].register_var("f", bcObj)
            self.grids[i].register_var("r", bcObj)

            self.grids[i].create()

            if self.verbose:
                print self.grids[i]        

            nx_t = nx_t*2
            ny_t = ny_t*2

            i += 1


        # provide coordinate and indexing information for the solution mesh
        solnGrid = self.grids[self.nlevels-1].grid

        self.ilo = solnGrid.ilo
        self.ihi = solnGrid.ihi
        self.jlo = solnGrid.jlo
        self.jhi = solnGrid.jhi
        
        self.x  = solnGrid.x
        self.dx = solnGrid.dx

        self.x2d = solnGrid.x2d

        self.y  = solnGrid.y
        self.dy = solnGrid.dy   # note, dy = dx is assumed 

        self.y2d = solnGrid.y2d

        self.solnGrid = solnGrid

        # store the source norm
        self.sourceNorm = 0.0

        # after solving, keep track of the number of cycles taken, the
        # relative error from the previous cycle, and the residual error
        # (normalized to the source norm)
        self.numCycles = 0
        self.residualError = 1.e33
        self.relativeError = 1.e33

        # keep track of where we are in the V
        self.currentCycle = -1
        self.currentLevel = -1
        self.upOrDown = ""

        # for visualization -- what frame are we outputting?
        self.vis = vis
        self.vis_title = vis_title
        self.frame = 0

    # these draw functions are for visualization purposes and are
    # not ordinarily used, except for plotting the progression of the
    # solution within the V
    def draw_V(self):

        xdown = numpy.linspace(0.0, 0.5, self.nlevels)
        xup = numpy.linspace(0.5, 1.0, self.nlevels)

        ydown = numpy.linspace(1.0, 0.0, self.nlevels)
        yup = numpy.linspace(0.0, 1.0, self.nlevels)

        pylab.plot(xdown, ydown, lw=2, color="k")
        pylab.plot(xup, yup, lw=2, color="k")

        pylab.scatter(xdown, ydown, marker="o", color="k", s=40)
        pylab.scatter(xup, yup, marker="o", color="k", s=40)

        if self.upOrDown == "down":
            pylab.scatter(xdown[self.nlevels-self.currentLevel-1], ydown[self.nlevels-self.currentLevel-1], 
                          marker="o", color="r", zorder=100, s=38)

        else:
            pylab.scatter(xup[self.currentLevel], yup[self.currentLevel], 
                          marker="o", color="r", zorder=100, s=38)

        pylab.text(0.7, 0.1, "V-cycle %d" % (self.currentCycle))
        pylab.axis("off")


    def draw_solution(self):
        
        myg = self.grids[self.currentLevel].grid

        v = self.grids[self.currentLevel].get_var("v")

        pylab.imshow(numpy.transpose(v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]),
                     interpolation="nearest", origin="lower",
                     extent=[self.xmin, self.xmax, self.ymin, self.ymax])

        #pylab.xlabel("x")
        pylab.ylabel("y")
        

        if self.currentLevel == self.nlevels-1:
            pylab.title(r"solving $L\phi = f$")
        else:
            pylab.title(r"solving $Le = r$")

        formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
        cb = pylab.colorbar(format=formatter, shrink=0.5)
    
        cb.ax.yaxis.offsetText.set_fontsize("small")
        cl = pylab.getp(cb.ax, 'ymajorticklabels')
        pylab.setp(cl, fontsize="small")


    def draw_main_solution(self):
        
        myg = self.grids[self.nlevels-1].grid

        v = self.grids[self.nlevels-1].get_var("v")

        pylab.imshow(numpy.transpose(v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]),
                     interpolation="nearest", origin="lower",
                     extent=[self.xmin, self.xmax, self.ymin, self.ymax])

        pylab.xlabel("x")
        pylab.ylabel("y")
        

        pylab.title(r"current fine grid solution")

        formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
        cb = pylab.colorbar(format=formatter, shrink=0.5)
    
        cb.ax.yaxis.offsetText.set_fontsize("small")
        cl = pylab.getp(cb.ax, 'ymajorticklabels')
        pylab.setp(cl, fontsize="small")


    def draw_main_error(self):
        
        myg = self.grids[self.nlevels-1].grid

        v = self.grids[self.nlevels-1].get_var("v")

        e = v - self.trueFunc(myg.x2d, myg.y2d)

        pylab.imshow(numpy.transpose(e[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]),
                     interpolation="nearest", origin="lower",
                     extent=[self.xmin, self.xmax, self.ymin, self.ymax])

        pylab.xlabel("x")
        pylab.ylabel("y")
        

        pylab.title(r"current fine grid error")

        formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
        cb = pylab.colorbar(format=formatter, shrink=0.5)
    
        cb.ax.yaxis.offsetText.set_fontsize("small")
        cl = pylab.getp(cb.ax, 'ymajorticklabels')
        pylab.setp(cl, fontsize="small")

    

    def get_solution(self):
        v = self.grids[self.nlevels-1].get_var("v")
        return v.copy()
        

    def get_solution_object(self):
        my_data = self.grids[self.nlevels-1]
        return my_data


    def init_solution(self, data):
        """
        initialize the solution to the elliptic problem by passing in
        a value for all defined zones
        """
        v = self.grids[self.nlevels-1].get_var("v")
        v[:,:] = data.copy()

        self.initializedSolution = 1


    def init_zeros(self):
        """
        set the initial solution to zero
        """
        v = self.grids[self.nlevels-1].get_var("v")
        v[:,:] = 0.0

        self.initializedSolution = 1


    def init_RHS(self, data):
        f = self.grids[self.nlevels-1].get_var("f")
        f[:,:] = data.copy()

        # store the source norm
        self.sourceNorm = error(self.grids[self.nlevels-1].grid, f)

        if self.verbose:
            print "Source norm = ", self.sourceNorm

        # note: if we wanted to do inhomogeneous Dirichlet BCs, we 
        # would modify the source term, f, here to include a boundary
        # charge

        self.initializedRHS = 1
        

    def compute_residual(self, level):
        """ compute the residual and store it in the r variable"""

        v = self.grids[level].get_var("v")
        f = self.grids[level].get_var("f")
        r = self.grids[level].get_var("r")

        myg = self.grids[level].grid

        # compute the residual 
        # r = f - alpha phi + beta L phi
        r[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
            f[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] - \
            self.alpha*v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] + \
            self.beta*( 
            (v[myg.ilo-1:myg.ihi  ,myg.jlo  :myg.jhi+1] + 
             v[myg.ilo+1:myg.ihi+2,myg.jlo  :myg.jhi+1] - 
             2.0*v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/(myg.dx*myg.dx) + 
            (v[myg.ilo  :myg.ihi+1,myg.jlo-1:myg.jhi  ] +
             v[myg.ilo  :myg.ihi+1,myg.jlo+1:myg.jhi+2] -
             2.0*v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])/(myg.dy*myg.dy) )

        
    def smooth(self, level, nsmooth):
        """ use Gauss-Seidel iterations to smooth """
        v = self.grids[level].get_var("v")
        f = self.grids[level].get_var("f")

        myg = self.grids[level].grid

        self.grids[level].fill_BC("v")

        # do red-black G-S
        i = 0
        while (i < nsmooth):

            xcoeff = self.beta/myg.dx**2
            ycoeff = self.beta/myg.dy**2

            # do the red black updating in four decoupled groups
            v[myg.ilo:myg.ihi+1:2,myg.jlo:myg.jhi+1:2] = \
                (f[myg.ilo:myg.ihi+1:2,myg.jlo:myg.jhi+1:2] +
                 xcoeff*(v[myg.ilo+1:myg.ihi+2:2,myg.jlo  :myg.jhi+1:2] +
                         v[myg.ilo-1:myg.ihi  :2,myg.jlo  :myg.jhi+1:2]) +
                 ycoeff*(v[myg.ilo  :myg.ihi+1:2,myg.jlo+1:myg.jhi+2:2] +
                         v[myg.ilo  :myg.ihi+1:2,myg.jlo-1:myg.jhi  :2])) / \
                (self.alpha + 2.0*xcoeff + 2.0*ycoeff)

            v[myg.ilo+1:myg.ihi+1:2,myg.jlo+1:myg.jhi+1:2] = \
                (f[myg.ilo+1:myg.ihi+1:2,myg.jlo+1:myg.jhi+1:2] +
                 xcoeff*(v[myg.ilo+2:myg.ihi+2:2,myg.jlo+1:myg.jhi+1:2] +
                         v[myg.ilo  :myg.ihi  :2,myg.jlo+1:myg.jhi+1:2]) +
                 ycoeff*(v[myg.ilo+1:myg.ihi+1:2,myg.jlo+2:myg.jhi+2:2] +
                         v[myg.ilo+1:myg.ihi+1:2,myg.jlo  :myg.jhi  :2])) / \
                (self.alpha + 2.0*xcoeff + 2.0*ycoeff)
            
            self.grids[level].fill_BC("v")
                                                     
            v[myg.ilo+1:myg.ihi+1:2,myg.jlo:myg.jhi+1:2] = \
                (f[myg.ilo+1:myg.ihi+1:2,myg.jlo:myg.jhi+1:2] +
                 xcoeff*(v[myg.ilo+2:myg.ihi+2:2,myg.jlo  :myg.jhi+1:2] +
                         v[myg.ilo  :myg.ihi  :2,myg.jlo  :myg.jhi+1:2]) +
                 ycoeff*(v[myg.ilo+1:myg.ihi+1:2,myg.jlo+1:myg.jhi+2:2] +
                         v[myg.ilo+1:myg.ihi+1:2,myg.jlo-1:myg.jhi  :2])) / \
                (self.alpha + 2.0*xcoeff + 2.0*ycoeff)

            v[myg.ilo:myg.ihi+1:2,myg.jlo+1:myg.jhi+1:2] = \
                (f[myg.ilo:myg.ihi+1:2,myg.jlo+1:myg.jhi+1:2] +
                 xcoeff*(v[myg.ilo+1:myg.ihi+2:2,myg.jlo+1:myg.jhi+1:2] +
                         v[myg.ilo-1:myg.ihi  :2,myg.jlo+1:myg.jhi+1:2]) +
                 ycoeff*(v[myg.ilo  :myg.ihi+1:2,myg.jlo+2:myg.jhi+2:2] +
                         v[myg.ilo  :myg.ihi+1:2,myg.jlo  :myg.jhi  :2])) / \
                (self.alpha + 2.0*xcoeff + 2.0*ycoeff)


            self.grids[level].fill_BC("v")

            if self.vis == 1:
                pylab.clf()

                pylab.subplot(221)
                self.draw_solution()

                pylab.subplot(222)        
                self.draw_V()

                pylab.subplot(223)        
                self.draw_main_solution()

                pylab.subplot(224)        
                self.draw_main_error()


                pylab.suptitle(self.vis_title, fontsize=18)

                pylab.draw()
                pylab.savefig("mg_%4.4d.png" % (self.frame))
                self.frame += 1

                                                     
            i += 1



    def solve(self, rtol = 1.e-11):

        # start by making sure that we've initialized the solution
        # and the RHS
        if not self.initializedSolution or not self.initializedRHS:
            msg.fail("ERROR: solution and RHS are not initialized")

        # for now, we will just do V-cycles, continuing until we
        # achieve the L2 norm of the relative solution difference is <
        # rtol
        if self.verbose:
            print "source norm = ", self.sourceNorm
            
        oldSolution = self.grids[self.nlevels-1].get_var("v").copy()
        
        converged = 0
        cycle = 1

        while (not converged and cycle <= self.maxCycles):

            self.currentCycle = cycle

            # zero out the solution on all but the finest grid
            level = 0
            while (level < self.nlevels-1):
                v = self.grids[level].zero("v")
                level += 1            

            # descending part
            if self.verbose:
                print "<<< beginning V-cycle (cycle %d) >>>\n" % cycle

            level = self.nlevels-1
            while (level > 0):

                self.currentLevel = level
                self.upOrDown = "down"

                fP = self.grids[level]
                cP = self.grids[level-1]

                # access to the residual
                r = fP.get_var("r")

                if self.verbose:
                    self.compute_residual(level)

                    print "  level = %d, nx = %d, ny = %d" %  \
                        (level, fP.grid.nx, fP.grid.ny)

                    print "  before G-S, residual L2 norm = %g" % \
                          (error(fP.grid, r) )
            
                # smooth on the current level
                self.smooth(level, self.nsmooth)

            
                # compute the residual
                self.compute_residual(level)

                if self.verbose:
                    print "  after G-S, residual L2 norm = %g\n" % \
                          (error(fP.grid, r) )


                # restrict the residual down to the RHS of the coarser level
                f_coarse = cP.get_var("f")
                f_coarse[:,:] = fP.restrict("r")

                level -= 1


            # solve the discrete coarse problem.  We could use any 
            # number of different matrix solvers here (like CG), but
            # since we are 2x2 by design at this point, we will just
            # smooth
            if self.verbose:
                print "  bottom solve:"

            self.currentLevel = 0

            bP = self.grids[0]

            if self.verbose:
                print "  level = %d, nx = %d, ny = %d\n" %  \
                    (level, bP.grid.nx, bP.grid.ny)

            self.smooth(0, self.nbottomSmooth)

            bP.fill_BC("v")

            
            # ascending part
            level = 1
            while (level < self.nlevels):

                self.currentLevel = level
                self.upOrDown = "up"

                fP = self.grids[level]
                cP = self.grids[level-1]

                # prolong the error up from the coarse grid
                e = cP.prolong("v")

                # correct the solution on the current grid
                v = fP.get_var("v")
                v += e

                if self.verbose:
                    self.compute_residual(level)
                    r = fP.get_var("r")

                    print "  level = %d, nx = %d, ny = %d" % \
                        (level, fP.grid.nx, fP.grid.ny)

                    print "  before G-S, residual L2 norm = %g" % \
                          (error(fP.grid, r) )
            
                # smooth
                self.smooth(level, self.nsmooth)

                if self.verbose:
                    self.compute_residual(level)

                    print "  after G-S, residual L2 norm = %g\n" % \
                          (error(fP.grid, r) )
            
                level += 1

            # compute the error with respect to the previous solution
            # this is for diagnostic purposes only -- it is not used to
            # determine convergence
            solnP = self.grids[self.nlevels-1]

            diff = (solnP.get_var("v") - oldSolution)/ \
                (solnP.get_var("v") + self.small)

            relativeError = error(solnP.grid, diff)

            oldSolution = solnP.get_var("v").copy()

            # compute the residual error, relative to the source norm
            self.compute_residual(self.nlevels-1)
            r = fP.get_var("r")

            if self.sourceNorm != 0.0:
                residualError = error(fP.grid, r)/self.sourceNorm
            else:
                residualError = error(fP.grid, r)

                
            if residualError < rtol:
                converged = 1
                self.numCycles = cycle
                self.relativeError = relativeError
                self.residualError = residualError
                fP.fill_BC("v")
                
            if self.verbose:
                print "cycle %d: relative err = %g, residual err = %g\n" % \
                      (cycle, relativeError, residualError)
            
            cycle += 1



        
        
