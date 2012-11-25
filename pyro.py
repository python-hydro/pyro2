#!/usr/bin/env python

import sys
import getopt
import numpy
import pylab
import time
from util import profile
from util import runparams
import os
from util import msg

usage = """
       usage:

      ./pyro [options] <solver name> <problem name> <input file>
"""


msg.bold('pyro ...')

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
    msg.fail("invalid calling sequence")
    sys.exit(2)

for o, a in opts:

    if o == "-i":
        print "-i passed"

try: solverName = next[0]
except IndexError:
    print usage
    msg.fail("ERROR: solver name not specified on command line")

try: problemName = next[1]
except IndexError:
    print usage
    msg.fail("ERROR: problem name not specified on command line")

try: paramFile = next[2]
except IndexError:
    print usage
    msg.fail("ERROR: parameter file not specified on command line")


otherCmds = []
if (len(next) > 3):
    otherCmds = next[3:]

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
        msg.fail("ERROR: inputs file does not exist")

runparams.LoadParams(paramFile, noNew=1)

# and any commandline overrides
runparams.CommandLineParams(otherCmds)

# write out the inputs.auto
runparams.PrintParamFile()


#-----------------------------------------------------------------------------
# initialization
#-----------------------------------------------------------------------------

# initialize the grid structure
myGrid, myData = solver.initialize()
    

# initialize the data
exec 'from ' + solverName + '.problems import *'

exec problemName + '.initData(myData)'


#-----------------------------------------------------------------------------
# pre-evolve
#-----------------------------------------------------------------------------
solver.preevolve(myData)


#-----------------------------------------------------------------------------
# evolve
#-----------------------------------------------------------------------------
tmax = runparams.getParam("driver.tmax")
max_steps = runparams.getParam("driver.max_steps")

init_tstep_factor = runparams.getParam("driver.init_tstep_factor")
max_dt_change = runparams.getParam("driver.max_dt_change")
fix_dt = runparams.getParam("driver.fix_dt")

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
    if (fix_dt > 0.0):
        dt = fix_dt
    else:
        if (n == 0):
            dt = init_tstep_factor*dt
            dtOld = dt
        else:
            dt = min(max_dt_change*dtOld, dt)
            dtOld = dt
            
    if (myData.t + dt > tmax):
        dt = tmax - myData.t

    # evolve for a single timestep
    solver.evolve(myData, dt)


    # increment the time
    myData.t += dt
    n += 1
    print "%5d %10.5f %10.5f" % (n, myData.t, dt)


    # output
    dt_out = runparams.getParam("io.dt_out")
    n_out = runparams.getParam("io.n_out")

    if (myData.t >= (nout + 1)*dt_out or n%n_out == 0):

        pfc = profile.timer("output")
        pfc.begin()

        msg.warning("outputting...")
        basename = runparams.getParam("io.basename")
        myData.write(basename + "%4.4d" % (n))
        nout += 1

        pfc.end()


    # visualization
    if (dovis): 
        pfd = profile.timer("vis")
        pfd.begin()

        solver.dovis(myData, n)
        store = runparams.getParam("vis.store_images")

        if (store == 1):
            basename = runparams.getParam("io.basename")
            pylab.savefig(basename + "%4.4d" % (n) + ".png")

        pfd.end()

pf.end()

runparams.printUnusedParams()
profile.timeReport()
