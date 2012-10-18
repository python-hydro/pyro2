#!/usr/bin/env python

"""

an example of using the multigrid class to solve Laplace's equation.  Here, we
solve

u_xx + u_yy = -2[(1-6x**2)y**2(1-y**2) + (1-6y**2)x**2(1-x**2)]
u = 0 on the boundary

this is the example from page 64 of the book `A Multigrid Tutorial, 2nd Ed.'

The analytic solution is u(x,y) = (x**2 - x**4)(y**4 - y**2)

"""
#from io import *
from numarray import *
from mesh import *
from util import compare

# the analytic solution
def true(x,y):
    return (x**2 - x**4)*(y**4 - y**2)


# the L2 error norm
def error(imin, imax, dx, jmin, jmax, dy, r):

    # L2 norm of elements in r, multiplied by dx to
    # normalize
    return sqrt(dx*dy*sum(r[imin:imax+1,jmin:jmax+1].flat**2))


# the righthand side
def f(x,y):
    return -2.0*((1.0-6.0*x**2)*y**2*(1.0-y**2) + (1.0-6.0*y**2)*x**2*(1.0-x**2))

                
# test the multigrid solver
nx = 128
ny = 128

# set the boundary conditions
xlbc = 0.0
xrbc = 0.0

ylbc = 0.0
yrbc = 0.0

# create the multigrid object
a = multigrid.MGfv(nx, ny,
                   xlBCtype="dirichlet", ylBCtype="dirichlet",
                   xrBCtype="dirichlet", yrBCtype="dirichlet",
                   xlBC = xlbc, xrBC = xrbc, ylBC = ylbc, yrBC = yrbc,
                   verbose=1)

# initialize the solution to 0
a.initSolution(zeros((nx, ny), Float64))

# initialize the RHS using the function f
rhs = f(a.x2d[a.imin:a.imax+1,a.jmin:a.jmax+1],
        a.y2d[a.imin:a.imax+1,a.jmin:a.jmax+1])
a.initRHS(rhs)

# solve to a relative tolerance of 1.e-11
a.solve(rtol=1.e-11)

# alternately, we can just use smoothing by uncommenting the following
#a.smooth(a.nlevels-1,5000)

# get the solution 
v = a.getSolution()

# compute the true solution from the analytic expression
b = true(a.x2d,a.y2d)

# compute the error
e = v - b

print "L2 error = %g, rel. error from previous cycle = %g, num. cycles = %d" % \
      (error(a.imin,a.imax, a.dx, a.jmin, a.jmax, a.dy, e), a.relativeError, a.numCycles)

# write out the finest level mesh object to a file
solnPatch = a.getSolutionPatch()
#io.writeFile("multigrid.test", solnPatch)


# compare to the stored solution
#testFile = "test/multigrid.test"
#testPatch = io.readFile(testFile)

print "comparing current solution to " + testFile

#compare.mesh2d(solnPatch,testPatch)


