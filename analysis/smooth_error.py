#!/usr/bin/env python

import numpy
import mesh.patch as patch
import getopt
import sys
import advection.problems.smooth as smooth

usage = """
      compare the output in file from the smooth advection problem to
      the analytic solution.

      usage: ./smooth_compare.py file
"""

def abort(string):
    print string
    sys.exit(2)


if not len(sys.argv) == 2:
    print usage
    sys.exit(2)


try: file1 = sys.argv[1]
except:
    print usage
    sys.exit(2)

myg, myd = patch.read(file1)


# create a new data object on the same grid
analytic = patch.ccData2d(myg, dtype=numpy.float64)

bco = myd.BCs[myd.vars[0]]
analytic.registerVar("density", bco)
analytic.create()

# use the original initialization routine to set the analytic solution
smooth.initData(analytic)

# compare the error
dens_numerical = myd.getVarPtr("density")
dens_analytic = analytic.getVarPtr("density")

print "mesh details"
print myg

aerr = abs(dens_numerical[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] -
            dens_analytic[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1])

rerr = aerr/dens_analytic[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]

# note that the numpy norm does not normalize by the number of elements, 
# so we explicitly do so here
l2a = numpy.sqrt(numpy.sum(aerr**2)/(myg.nx*myg.ny))
l2r = numpy.sqrt(numpy.sum(rerr**2)/(myg.nx*myg.ny))
print "error norms (absolute, relative): ", l2a, l2r








