"""

The multigrid module provides a framework for solving elliptic
problems.  A multigrid object is just a list of grids, from the finest
mesh down (by factors of two) to a single interior zone (each grid
has the same number of guardcells).

The main multigrid class (MGfv) is setup to solve Poisson's equation
with Dirichlet boundary conditions.  Implementations for other
equations can be created by making a new class that inherits the MGfv
class, and provides new versions of the residual and smooth methods.

The general usage is as follows:

> a = multigrid.MGfv(nx, ny, xlBC=0, xrBC=1, ylBC=0, yrBC=1, verbose=1)

this creates the multigrid object a, with a finest grid of nx by ny
zones and a left/right x/y boundary conditions of 0 and 1 respectively.
Setting verbose = 1 causing debugging information to be output, so you
can see the residual errors in each of the V-cycles.

> a.initSolution(zeros((nx, ny), Float64))

this initializes the solution vector with zeros

> a.initRHS(zeros((nx, ny), Float64))

this initializes the RHS on the finest grid to 0 (Laplace's equation).
Any RHS can be set by passing through an array of nx values here.

Then to solve, you just do:

> a.solve(rtol = 1.e-10)

where rtol is the desired tolerance (relative difference in solution from
one cycle to the next).

to access the final solution, use the getSolution method

v = a.getSolution()

For convenience, the grid information on the solution level is available as
attributes to the class,

a.imin, a.imax, a.jmin, a.jmax are the indices bounding the interior
of the solution array (i.e. excluding the guardcells).

a.x and a.y are the coordinate arrays
a.dx and a.dy are the grid spacings

"""

import patch
import math
from numarray import *

DIRICHLET = 1
NEUMANN = 2
PERIODIC = 3

def error(imin, imax, dx, jmin, jmax, dy, r):

    # L2 norm of elements in r, multiplied by dx*dy to
    # normalize
    #return sqrt(dx*dy*sum(r[imin:imax+1,jmin:jmax+1].flat**2))
    return max(r[imin:imax+1,jmin:jmax+1].flat)

class MGfv:
    """ presently, the MG class requires nx = ny be a power of 2 and dx = dy,
        for simplicity """
    
    def __init__(self, nx, ny, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
                 xlBCtype="dirichlet", xrBCtype="dirichlet",
                 ylBCtype="dirichlet", yrBCtype="dirichlet",
                 xlBC=0, xrBC=0, ylBC=0, yrBC=0,
                 verbose=0):


        if (nx != ny):
            print "ERROR: multigrid currently requires nx = ny"
            return -1
        
        self.nx = nx
        self.ny = ny        
        
        self.ng = 3

        self.xmin = xmin
        self.xmax = xmax

        self.ymin = ymin
        self.ymax = ymax        

        if ((xmax-xmin) != (ymax-ymin)):
            print "ERROR: multigrid currently requires a square domain"
            return -1
        
        self.xlBC = xlBC
        self.xrBC = xrBC

        self.ylBC = ylBC
        self.yrBC = yrBC        

        if xlBCtype == "dirichlet":
            self.xlBCtype_i = DIRICHLET
        elif xlBCtype == "neumann":
            self.xlBCtype_i = NEUMANN
        elif xlBCtype == "periodic":
            self.xlBCtype_i = PERIODIC
            
        if xrBCtype == "dirichlet":
            self.xrBCtype_i = DIRICHLET
        elif xrBCtype == "neumann":
            self.xrBCtype_i = NEUMANN
        elif xrBCtype == "periodic":
            self.xrBCtype_i = PERIODIC

        if ylBCtype == "dirichlet":
            self.ylBCtype_i = DIRICHLET
        elif ylBCtype == "neumann":
            self.ylBCtype_i = NEUMANN
        elif ylBCtype == "periodic":
            self.ylBCtype_i = PERIODIC

        if yrBCtype == "dirichlet":
            self.yrBCtype_i = DIRICHLET
        elif yrBCtype == "neumann":
            self.yrBCtype_i = NEUMANN
        elif yrBCtype == "periodic":
            self.yrBCtype_i = PERIODIC
        
        self.nsmooth = 2

        self.maxCycles = 100
        
        self.verbose = verbose

        # a small number used in computing the error, so we don't divide by 0
        self.small = 1.e-16
        
        # keep track of whether we've initialized the solution
        self.initializedSolution = 0
        self.initializedRHS = 0
        
        # assume that self.nx = 2^nlevels and that nx = ny
        self.nlevels = int(math.log(self.nx)/math.log(2.0)) + 1

        # a multigrid object will be a list of grids
        self.grids = []

        # create the grids.  Here, self.grids[0] will be the coarsest
        # grid and self.grids[nlevel-1] will be the finest grid
        # we store the solution, v, the rhs, f.
        i = 0
        nx_t = ny_t = 1

        while (i < self.nlevels):
            self.grids.append(patch.cellCentered(nx_t, ny_t, ng=self.ng,
                                                 xmin=xmin, xmax=xmax,
                                                 ymin=ymin, ymax=ymax))
            self.grids[i].registerVar("v")
            self.grids[i].registerVar("f")
            self.grids[i].init()

            if self.verbose:
                print self.grids[i]        

            nx_t = nx_t*2
            ny_t = ny_t*2

            i += 1


        # provide coordinate and indexing information for the solution mesh
        self.imin = self.grids[self.nlevels-1].imin
        self.imax = self.grids[self.nlevels-1].imax
        self.jmin = self.grids[self.nlevels-1].jmin
        self.jmax = self.grids[self.nlevels-1].jmax        
        
        self.x = self.grids[self.nlevels-1].x
        self.dx = self.grids[self.nlevels-1].dx

        self.x2d = self.grids[self.nlevels-1].x2d

        self.y = self.grids[self.nlevels-1].y
        self.dy = self.grids[self.nlevels-1].dy   # note, dy = dx is assumed 

        self.y2d = self.grids[self.nlevels-1].y2d

        # store the source norm
        self.sourceNorm = 0.0

        # after solving, keep track of the number of cycles taken, the
        # relative error from the previous cycle, and the residual error
        # (normalized to the source norm)
        self.numCycles = 0
        self.residualError = 1.e33
        self.relativeError = 1.e33

    def fillBC(self, level):

        v = self.grids[level].getVar("v")

        imin = self.grids[level].imin
        imax = self.grids[level].imax

        jmin = self.grids[level].jmin
        jmax = self.grids[level].jmax        


        # inhomogeneous boundary conditions are converted into homogeneous
        # BCs with the addition of a boundary charge to the RHS, so here,
        # we just need to do homogeneous BCs.
        if self.xlBCtype_i == DIRICHLET:
            v[imin-1,:] = -v[imin,:]
        elif self.xlBCtype_i == NEUMANN:
            v[imin-1,:] = v[imin,:]
        elif self.xlBCtype_i == PERIODIC:
            v[imin-1,:] = v[imax,:]
            
        if self.xrBCtype_i == DIRICHLET:
            v[imax+1,:] = -v[imax,:]
        elif self.xrBCtype_i == NEUMANN:
            v[imax+1,:] = v[imax,:]
        elif self.xrBCtype_i == PERIODIC:
            v[imax+1,:] = v[imin,:]
        
            
        if self.ylBCtype_i == DIRICHLET:
            v[:,jmin-1] = -v[:,jmin]
        elif self.ylBCtype_i == NEUMANN:
            v[:,jmin-1] = v[:,jmin]
        elif self.ylBCtype_i == PERIODIC:
            v[:,jmin-1] = v[:,jmax]

        if self.yrBCtype_i == DIRICHLET:
            v[:,jmax+1] = -v[:,jmax]
        elif self.yrBCtype_i == NEUMANN:
            v[:,jmax+1] = v[:,jmax]
        elif self.yrBCtype_i == PERIODIC:
            v[:,jmax+1] = v[:,jmin]
    

    def getSolution(self):
        v = self.grids[self.nlevels-1].getVar("v")
        return v.copy()
        

    def getSolutionPatch(self):
        return self.grids[self.nlevels-1]
        

    def initSolution(self, data):
        v = self.grids[self.nlevels-1].getVar("v")
        v[self.imin:self.imax+1,self.jmin:self.jmax+1] = data

        self.initializedSolution = 1


    def initRHS(self, data):
        f = self.grids[self.nlevels-1].getVar("f")
        f[self.imin:self.imax+1,self.jmin:self.jmax+1] = data

        # store the source norm
        self.sourceNorm = error(self.imin, self.imax, self.dx,
                                self.jmin, self.jmax, self.dy, f)
        
        # given a left boundary condition, lBC, we want to find the
        # guardcell value that, together with the first interior zone
        # yield lBC on the boundary interface.
        #
        #    .      |      .      .
        #    .      |      .      .
        #    +------+------+------+--
        #       -1     0       1
        #
        # Here, -1 is the first guardcell, 0 is the first interior zone,
        # and U = lBC on the interface between -1 and 0, so
        #
        # lBC = 0.5*(U_{-1} + U_0)
        #
        # therefore, U_{-1} = 2*lBC - U_0
        #
        # in the homogeneous case, it would just be U_{-1} = -U_0, therefore
        # we can bring the 2*lBC part over to the righthand side and treat
        # it as a boundary charge,
        #
        # The normal differencing is
        #
        # U_1 - 2*U_0 + U_{-1} = dx**2 f_0
        #
        # for homogeneous BCs, this would be
        #
        # U_1 - 2*U_0 - U_0 = dx**2 f_0
        #
        # for inhomogeneous, it is
        #
        # U_1 - 2*U_0 + (2*lBC - U_0) = dx**2 f_0
        #
        # or
        #
        # U_1 - 2*U_0 - U_0 = dx**2 * (f_0 - 2*lBC/dx**2)
        #
        # which looks like the homogeneous BC case with the addition of the
        # boundary charge, -2*lBC/dx**2.

        if self.xlBCtype_i == DIRICHLET:
            f[self.imin,self.jmin:self.jmax+1] += -2*self.xlBC/(self.dx*self.dx)

        if self.xrBCtype_i == DIRICHLET:
            f[self.imax,self.jmin:self.jmax+1] += -2*self.xrBC/(self.dx*self.dx)

        if self.ylBCtype_i == DIRICHLET:
            f[self.imin:self.imax+1,self.jmin] += -2*self.ylBC/(self.dy*self.dy)

        if self.yrBCtype_i == DIRICHLET:
            f[self.imin:self.imax+1,self.jmax] += -2*self.yrBC/(self.dy*self.dy)        


        # with these additions, we can treat the problem as having homogeneous
        # boundary conditions.

        self.initializedRHS = 1
        

    def residual(self, level):
        """ compute the residual """

        v = self.grids[level].getVar("v")
        f = self.grids[level].getVar("f")

        imin = self.grids[level].imin
        imax = self.grids[level].imax

        jmin = self.grids[level].jmin
        jmax = self.grids[level].jmax        

        nx = self.grids[level].nx
        ny = self.grids[level].ny        
        ng = self.grids[level].ng        

        dx = self.grids[level].dx
        dy = self.grids[level].dy        

        r = zeros((2*ng+nx, 2*ng+ny), Float64)

        # compute the residual (assuming dx = dy)
        r[imin:imax+1,jmin:jmax+1] = f[imin:imax+1,jmin:jmax+1] - \
                                     (v[imin-1:imax,jmin:jmax+1] +
                                      v[imin+1:imax+2,jmin:jmax+1] +
                                      v[imin:imax+1,jmin-1:jmax] +
                                      v[imin:imax+1,jmin+1:jmax+2] -
                                      4.0*v[imin:imax+1,jmin:jmax+1])/(dx*dx)

        return r
        
        
    def smooth(self, level, nsmooth):

        v = self.grids[level].getVar("v")
        f = self.grids[level].getVar("f")

        imin = self.grids[level].imin
        imax = self.grids[level].imax

        jmin = self.grids[level].jmin
        jmax = self.grids[level].jmax        

        dx = self.grids[level].dx
        dy = self.grids[level].dy        

        self.fillBC(level)

        # assume we are solving u_{i+1} - 2 u_i + u_{i-1} = dx**2 f
        # do red-black G-S
        i = 0
        while (i < nsmooth):

            # do the red black updating in four decoupled groups
            v[imin:imax+1:2,jmin:jmax+1:2] = 0.25*(v[imin-1:imax:2,jmin:jmax+1:2] +
                                                   v[imin+1:imax+2:2,jmin:jmax+1:2] +
                                                   v[imin:imax+1:2,jmin-1:jmax:2] +
                                                   v[imin:imax+1:2,jmin+1:jmax+2:2] -
                                                   dx*dx*f[imin:imax+1:2,jmin:jmax+1:2])


            v[imin+1:imax+1:2,jmin+1:jmax+1:2] = 0.25*(v[imin:imax:2,jmin+1:jmax+1:2] +
                                                       v[imin+2:imax+2:2,jmin+1:jmax+1:2] +
                                                       v[imin+1:imax+1:2,jmin:jmax:2] +
                                                       v[imin+1:imax+1:2,jmin+2:jmax+2:2] -
                                                       dx*dx*f[imin+1:imax+1:2,jmin+1:jmax+1:2])
            
            self.fillBC(level)                                                     
            v[imin+1:imax+1:2,jmin:jmax+1:2] = 0.25*(v[imin:imax:2,jmin:jmax+1:2] +
                                                     v[imin+2:imax+2:2,jmin:jmax+1:2] +
                                                     v[imin+1:imax+1:2,jmin-1:jmax:2] +
                                                     v[imin+1:imax+1:2,jmin+1:jmax+2:2] -
                                                     dx*dx*f[imin+1:imax+1:2,jmin:jmax+1:2])


            v[imin:imax+1:2,jmin+1:jmax+1:2] = 0.25*(v[imin-1:imax:2,jmin+1:jmax+1:2] +
                                                     v[imin+1:imax+2:2,jmin+1:jmax+1:2] +
                                                     v[imin:imax+1:2,jmin:jmax:2] +
                                                     v[imin:imax+1:2,jmin+2:jmax+2:2] -
                                                     dx*dx*f[imin:imax+1:2,jmin+1:jmax+1:2])

            self.fillBC(level)                                                     
            i += 1



    def solve(self, rtol = 1.e-11):

        # start by making sure that we've initialized the solution
        # and the RHS
        if (not self.initializedSolution or not self.initializedRHS):
            print "ERROR: solution and RHS are not initialized"
            return -1

        # for now, we will just do V-cycles, continuing until we achieve the L2 
        # norm of the relative solution difference is < rtol
        if self.verbose:
            print "source norm = ", self.sourceNorm
            
        oldSolution = self.grids[self.nlevels-1].getVar("v").copy()
        
        converged = 0
        cycle = 1
        while (not converged and cycle <= self.maxCycles):

            # zero out the solution on all but the finest grid
            level = 0
            while (level < self.nlevels-1):
                v = self.grids[level].getVar("v")
                v[:] = 0.0
                
                level += 1            

            # descending part
            if self.verbose:
                print "<<< beginning V-cycle (cycle %d) >>>\n" % cycle

            level = self.nlevels-1
            while (level > 0):

                fP = self.grids[level]
                cP = self.grids[level-1]

                if self.verbose:
                    print "  level = %d, nx = %d, ny = %d" % (level, fP.nx, fP.ny)
                    print "  before G-S, residual L2 norm = %g" % \
                          (error(fP.imin, fP.imax, fP.dx,
                                 fP.jmin, fP.jmax, fP.dy, self.residual(level)))
            
                # smooth on the current level
                self.smooth(level, self.nsmooth)

                if self.verbose:
                    print "  after G-S, residual L2 norm = %g\n" % \
                          (error(fP.imin, fP.imax, fP.dx,
                                 fP.jmin, fP.jmax, fP.dy, self.residual(level)))
            
                # compute the residual
                r = self.residual(level)

                # restrict the residual down to the RHS of the coarser level
                patch._restrict_cc(fP.nx, fP.ny, fP.ng, r, cP.nx, cP.ny, cP.ng, cP.getVar("f"))

                level -= 1

            # solve the discrete coarse problem exactly
            if self.verbose:
                print "  <<< bottom solve >>>\n"

            bP = self.grids[0]
            v = bP.getVar("v")
            f = bP.getVar("f")        

            v[bP.imin] = -0.125*f[bP.imin]*bP.dx*bP.dx

            self.fillBC(level)

            
            # ascending part
            level = 1
            while (level < self.nlevels):

                fP = self.grids[level]
                cP = self.grids[level-1]

                # allocate storage for the error
                e = zeros((2*fP.ng+fP.nx, 2*fP.ng+fP.ny), Float64)

                # prolong the error up from the coarse grid
                patch._prolong_cc(cP.nx, cP.ny, cP.ng, cP.getVar("v"), fP.nx, fP.ny, fP.ng, e)

                # correct the solution on the current grid
                v = fP.getVar("v")
                v += e

                if self.verbose:
                    print "  level = %d, nx = %d, ny = %d" % (level, fP.nx, fP.ny)
                    print "  before G-S, residual L2 norm = %g" % \
                          (error(fP.imin, fP.imax, fP.dx,
                                 fP.jmin, fP.jmax, fP.dy, self.residual(level)))
            
                # smooth
                self.smooth(level, self.nsmooth)

                if self.verbose:
                    print "  after G-S, residual L2 norm = %g\n" % \
                          (error(fP.imin, fP.imax, fP.dx,
                                 fP.jmin, fP.jmax, fP.dy, self.residual(level)))
            
                level += 1

            # compute the error with respect to the previous solution
            # this is for diagnostic purposes only -- it is not used to
            # determine convergence
            solnP = self.grids[self.nlevels-1]

            diff = (solnP.getVar("v") - oldSolution)/(solnP.getVar("v") + self.small)
            relativeError = error(solnP.imin, solnP.imax, solnP.dx,
                        solnP.jmin, solnP.jmax, solnP.dy, diff)

            oldSolution = solnP.getVar("v").copy()

            # compute the residual error, relative to the source norm
            if (self.sourceNorm != 0.0):
                residualError = \
                              error(fP.imin, fP.imax, fP.dx, \
                                    fP.jmin, fP.jmax, fP.dy, \
                                    self.residual(self.nlevels-1))/  \
                                    self.sourceNorm
            else:
                residualError = \
                              error(fP.imin, fP.imax, fP.dx, \
                                    fP.jmin, fP.jmax, fP.dy, \
                                    self.residual(self.nlevels-1))

                
            if (residualError < rtol):
                converged = 1
                self.numCycles = cycle
                self.relativeError = relativeError
                self.residualError = residualError
                self.fillBC(self.nlevels-1)
                
            if self.verbose:
                print "cycle %d: relative error = %g, residual error = %g\n" % \
                      (cycle, relativeError, residualError)
            
            cycle += 1
            


        
        
