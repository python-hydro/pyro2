#!/usr/bin/env python

import numpy
import mesh.patch
import getopt
import sys

usage = """
      usage: ./compare.py file1 file2
"""

def abort(string):
    print string
    sys.exit(2)


if not len(sys.argv) == 3:
    print usage
    sys.exit(2)


file1 = sys.argv[1]
file2 = sys.argv[2]

myg1, myd1 = mesh.patch.read(file1)
myg2, myd2 = mesh.patch.read(file2)


# compare the grids
if (not myg1 == myg2): abort("ERROR: grids don't agree")

# compare the data
if (not myd1.vars == myd2.vars): abort("ERROR: variable lists doesn't agree")

print " "
print "grids agree"

print " "
print "variable comparisons:"

n = 0
while (n < myd1.nvar):

    d1 = myd1.getVarPtr(myd1.vars[n])
    d2 = myd2.getVarPtr(myd2.vars[n])

    err = numpy.max(numpy.abs(d1[myg1.ilo:myg1.ihi+1,myg1.jlo:myg1.jhi+1] -
                              d2[myg2.ilo:myg2.ihi+1,myg2.jlo:myg2.jhi+1]))

    print "%20s error = %20.10g" % (myd1.vars[n], err)

    n += 1



