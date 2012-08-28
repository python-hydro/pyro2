#!/usr/bin/env python

import sys
import getopt
import numpy
import pylab

from util import runparams

usage = """
       usage:

      ./pyro [options] <solver name> <problem name> <input file>
"""


print 'pyro ...'

#-----------------------------------------------------------------------------
# command line arguments / solver setup
#-----------------------------------------------------------------------------

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

try: problemName = next[1]
except IndexError:
    print 'ERROR: problem name not specified on command line'
    print usage
    sys.exit(2)

try: paramFile = next[2]
except IndexError:
    print 'ERROR: parameter file not specified on command line'
    print usage
    sys.exit(2)



# actually import the solver-specific stuff under the 'solver' namespace
exec 'import ' + solverName + ' as solver'


#-----------------------------------------------------------------------------
# runtime parameters
#-----------------------------------------------------------------------------

# parameter defaults
runparams.LoadParams("_defaults")
runparams.LoadParams(solverName + "/_defaults")

# problem-specific runtime parameters
runparams.LoadParams(solverName + "/problems/_" + problemName + ".defaults")

# now read in the inputs file
runparams.LoadParams(paramFile)

runparams.PrintParamFile()



#-----------------------------------------------------------------------------
# initialization
#-----------------------------------------------------------------------------

# initialize the grid structure
myGrid, myData = solver.initialize()
    

# initialize the data
exec 'from ' + solverName + '.problems import *'

exec problemName + '.fillPatch(myData)'


# debugging -- look at the data
dens = myData.getVarPtr("density")
myg = myData.grid


#-----------------------------------------------------------------------------
# evolve
#-----------------------------------------------------------------------------
tmax = runparams.getParam("driver.tmax")

pylab.ion()

n = 0
t = 0.0
while (t < tmax):

    # fill boundary conditions
    myData.fillBC("density")


    # get the timestep
    dt = solver.timestep(myData)

    if (t + dt > tmax):
        dt = tmax - t

    solver.evolve(myData, dt)


    # increment the time
    t += dt
    n += 1
    print "%5d %10.5f %10.5f" % (n, t, dt)


    # output
    tplot = runparams.getParam("io.tplot")
    print dt, tplot, t % tplot
    if (t % tplot <= dt):

        print "outputting..."
        basename = runparams.getParam("io.basename")
        myData.write(basename + "%4.4d" % (n))


    # visualization
    solver.dovis(myData, n)
    


