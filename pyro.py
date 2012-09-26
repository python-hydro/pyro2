#!/usr/bin/env python

import sys
import getopt
import numpy
import pylab
import time
from util import profile
from util import runparams
import os

usage = """
       usage:

      ./pyro [options] <solver name> <problem name> <input file>
"""


print 'pyro ...'

pf = profile.timer("main")
pf.begin()

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
if (not os.path.isfile(paramFile)):
    # check if the param file lives in the solver's problems directory
    paramFile = solverName + "/problems/" + paramFile
    if (not os.path.isfile(paramFile)):
        print 'ERROR: inputs file does not exist'
        print paramFile
        sys.exit(2)

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



#-----------------------------------------------------------------------------
# evolve
#-----------------------------------------------------------------------------
tmax = runparams.getParam("driver.tmax")
max_steps = runparams.getParam("driver.max_steps")

pylab.ion()

n = 0
myData.t = 0.0

# output the 0th data
basename = runparams.getParam("io.basename")
myData.write(basename + "%4.4d" % (n))

dovis = runparams.getParam("vis.dovis")
if (dovis): 
    pylab.figure(num=1, figsize=(8,6), dpi=100, facecolor='w')
    solver.dovis(myData, 0)
    

nout = 0

while (myData.t < tmax and n < max_steps):

    # fill boundary conditions
    pfb = profile.timer("fillBC")
    pfb.begin()
    myData.fillBCAll()
    pfb.end()

    # get the timestep
    dt = solver.timestep(myData)

    if (myData.t + dt > tmax):
        dt = tmax - myData.t

    solver.evolve(myData, dt)


    # increment the time
    myData.t += dt
    n += 1
    print "%5d %10.5f %10.5f" % (n, myData.t, dt)


    # output
    
    tplot = runparams.getParam("io.tplot")
    if (myData.t >= (nout + 1)*tplot):

        pfc = profile.timer("output")
        pfc.begin()

        print "outputting..."
        basename = runparams.getParam("io.basename")
        myData.write(basename + "%4.4d" % (n))
        nout += 1

        pfc.end()


    # visualization
    if (dovis): 
        pfd = profile.timer("vis")
        pfd.begin()
        solver.dovis(myData, n)
        pfd.end()

pf.end()

profile.timeReport()
