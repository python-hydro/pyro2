#!/usr/bin/env python

import sys
import getopt
import numpy

from util import runparams

usage = """
       usage:

      ./pyro [options] <solver name> <problem name> <input file>
"""


print 'pyro ...'

# parse the runtime arguments.  We specify a solver (which we import
# locally under the namespace 'solver', the problem name, and the
# input file name

if len(sys.argv) == 1:
    print usage
    sys.exit(2)


try: opts, next = getopt.getopt(sys.argv[1:], "i")
except getopt.GetoptError:
    print "invalid calling sequence"
    sys.exit(2)

for o, a in opts:

    if o == "-i":
        print "-i passed"

try: solverName = next[0]
except IndexError:
    print "ERROR: solver name not specified on command line"
    print usage
    sys.exit(2)

# actually import the solver-specific stuff
exec 'import ' + solverName + ' as solver'

# parameter defaults
runparams.LoadParams(solverName + "/_defaults")

# problem-specific runtime parameters


# initialize the grid
myGrid, myData = solver.initialize()
    
