#!/usr/bin/env python

import numpy
import pylab
import sys
import mesh.patch as patch

# plot an output file using the solver's dovis script

def makeplot(myd, solverName):

    exec 'import ' + solverName + ' as solver'

    pylab.figure(num=1, figsize=(6.5,5.25), dpi=100, facecolor='w')

    solver.dovis(myd, 0)
    pylab.savefig("plot.png")
    pylab.show()


if __name__== "__main__":

    print sys.argv

    solver = sys.argv[1]
    file = sys.argv[2]

    myg, myd = patch.read(file)

    makeplot(myd, solver)




