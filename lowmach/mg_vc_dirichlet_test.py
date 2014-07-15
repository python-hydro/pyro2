#!/usr/bin/env python

"""
Test the variable-coefficient MG solver with Dirichlet boundary conditions.

Here we solve:

   div . ( alpha grad phi ) = f

with

   alpha = 2.0 + cos(2.0*pi*x)*cos(2.0*pi*y)

   f = -16.0*pi**2*(cos(2*pi*x)*cos(2*pi*y) + 1)*sin(2*pi*x)*sin(2*pi*y)

This has the exact solution:

   phi = sin(2.0*pi*x)*sin(2.0*pi*y)

on [0,1] x [0,1]

We use Dirichlet BCs on phi.  For alpha, we do not have to impose the
same BCs, since that may represent a different physical quantity.
Here we take alpha to have Neumann BCs.  (Dirichlet BCs for alpha will
force it to 0 on the boundary, which is not correct here)

"""

from __future__ import print_function

import sys

import numpy
import mesh.patch as patch
import variable_coeff_MG as MG
import pylab

pi = numpy.pi
sin = numpy.sin
cos = numpy.cos
exp = numpy.exp

# the analytic solution
def true(x,y):
    return sin(2.0*pi*x)*sin(2.0*pi*y)


# the coefficients
def alpha(x,y):
    return 2.0 + cos(2.0*pi*x)*cos(2.0*pi*y)


# the L2 error norm
def error(myg, r):

    # L2 norm of elements in r, multiplied by dx to
    # normalize
    return numpy.sqrt(myg.dx*myg.dy*numpy.sum((r[myg.ilo:myg.ihi+1,
                                                 myg.jlo:myg.jhi+1]**2).flat))


# the righthand side
def f(x,y):
    return -16.0*pi**2*(cos(2*pi*x)*cos(2*pi*y) + 1)*sin(2*pi*x)*sin(2*pi*y)

    
# test the multigrid solver
nx = 128
ny = nx


# create the coefficient variable
g = patch.Grid2d(nx, ny, ng=1)
d = patch.CellCenterData2d(g)
bc_c = patch.BCObject(xlb="neumann", xrb="neumann",
                      ylb="neumann", yrb="neumann")
d.register_var("c", bc_c)
d.create()

c = d.get_var("c")
c[:,:] = alpha(g.x2d, g.y2d)


pylab.clf()

pylab.figure(figsize=(5.0,5.0), dpi=100, facecolor='w')

pylab.imshow(numpy.transpose(c[g.ilo:g.ihi+1,g.jlo:g.jhi+1]), 
          interpolation="nearest", origin="lower",
          extent=[g.xmin, g.xmax, g.ymin, g.ymax])

pylab.xlabel("x")
pylab.ylabel("y")

pylab.title("nx = {}".format(nx))

pylab.colorbar()

pylab.savefig("mg_alpha.png")


# create the multigrid object
a = MG.VarCoeffCCMG2d(nx, ny,
                      xl_BC_type="dirichlet", yl_BC_type="dirichlet",
                      xr_BC_type="dirichlet", yr_BC_type="dirichlet",
                      nsmooth=10,
                      nsmooth_bottom=50,
                      coeffs=c, coeffs_bc=bc_c,
                      verbose=1, vis=0, true_function=true)


# debugging
# for i in range(a.nlevels):
#     print(i)
#     print(a.grids[i].get_var("coeffs"))
    


# initialize the solution to 0
a.init_zeros()

# initialize the RHS using the function f
rhs = f(a.x2d, a.y2d)
a.init_RHS(rhs)

# solve to a relative tolerance of 1.e-11
a.solve(rtol=1.e-11)
#a.smooth(a.nlevels-1, 50000)

# alternately, we can just use smoothing by uncommenting the following
#a.smooth(a.nlevels-1,50000)

# get the solution 
v = a.get_solution()

# compute the error from the analytic solution
b = true(a.x2d,a.y2d)
e = v - b

print(" L2 error from true solution = %g\n rel. err from previous cycle = %g\n num. cycles = %d" % \
      (error(a.soln_grid, e), a.relative_error, a.num_cycles))


# plot it
pylab.clf()

pylab.figure(figsize=(10.0,5.0), dpi=100, facecolor='w')


pylab.subplot(121)

pylab.imshow(numpy.transpose(v[a.ilo:a.ihi+1,a.jlo:a.jhi+1]), 
          interpolation="nearest", origin="lower",
          extent=[a.xmin, a.xmax, a.ymin, a.ymax])

pylab.xlabel("x")
pylab.ylabel("y")

pylab.title("nx = {}".format(nx))

pylab.colorbar()


pylab.subplot(122)

pylab.imshow(numpy.transpose(e[a.ilo:a.ihi+1,a.jlo:a.jhi+1]), 
          interpolation="nearest", origin="lower",
          extent=[a.xmin, a.xmax, a.ymin, a.ymax])

pylab.xlabel("x")
pylab.ylabel("y")

pylab.title("error")

pylab.colorbar()

pylab.tight_layout()

pylab.savefig("mg_test.png")


# store the output for later comparison
my_data = a.get_solution_object()
my_data.write("mg_test")


