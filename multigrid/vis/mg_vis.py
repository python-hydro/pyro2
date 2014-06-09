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
import numpy
import mesh.patch as patch
import multigrid_vis
import pylab

# the analytic solution
def true(x,y):
    return (x**2 - x**4)*(y**4 - y**2)


# the L2 error norm
def error(myg, r):

    # L2 norm of elements in r, multiplied by dx to
    # normalize
    return numpy.sqrt(myg.dx*myg.dy*numpy.sum((r[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]**2).flat))


# the righthand side
def f(x,y):
    return -2.0*((1.0-6.0*x**2)*y**2*(1.0-y**2) + (1.0-6.0*y**2)*x**2*(1.0-x**2))

                
# test the multigrid solver
nx = 64
ny = 64


# create the multigrid object
a = multigrid_vis.CellCenterMG2d(nx, ny,
                                 xlBCtype="dirichlet", ylBCtype="dirichlet",
                                 xrBCtype="dirichlet", yrBCtype="dirichlet",
                                 verbose=0, trueFunc=true)

pylab.ion()

pylab.figure(num=1, figsize=(12.8,7.2), dpi=100, facecolor='w')



# initialize the solution to 0
init = a.solnGrid.scratch_array()

a.initSolution(init)

# initialize the RHS using the function f
rhs = f(a.x2d, a.y2d)
a.initRHS(rhs)

# solve to a relative tolerance of 1.e-11
a.solve(rtol=1.e-11)

# alternately, we can just use smoothing by uncommenting the following
#a.smooth(a.nlevels-1,50000)

# get the solution 
v = a.getSolution()

# compute the error from the analytic solution
b = true(a.x2d,a.y2d)
e = v - b

print " L2 error from true solution = %g\n rel. err from previous cycle = %g\n num. cycles = %d" % \
      (error(a.solnGrid, e), a.relativeError, a.numCycles)


# plot it
#pylab.figure(num=1, figsize=(2.10,2.10), dpi=100, facecolor='w')
pylab.figure(num=1, figsize=(5.0,5.0), dpi=100, facecolor='w')

pylab.imshow(numpy.transpose(v[a.ilo:a.ihi+1,a.jlo:a.jhi+1]), 
          interpolation="nearest", origin="lower",
          extent=[a.xmin, a.xmax, a.ymin, a.ymax])

#pylab.axis("off")
#pylab.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=1.0)

pylab.xlabel("x")
pylab.ylabel("y")

pylab.savefig("mg_test.png")


# store the output for later comparison
my_data = a.getSolutionObjPtr()
my_data.write("mg_test")


