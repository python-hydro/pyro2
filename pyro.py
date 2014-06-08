#!/usr/bin/env python

import getopt
import os
import sys
import time

import numpy
import pylab

import compare
import mesh.patch as patch
from util import msg, profile, runparams

usage = """
       usage:

      ./pyro [options] <solver> <problem> <input file> [runtime parameters]

      <solver> is one of:
          advection
          compressible
          diffusion
          incompressible

      <problem> is one of the problems defined in the solver's problems/
      sub-directory

      <input file> is the inputs file to use for that problem.  You can
      refer to a file in the solver's problem directory directly, without
      giving the path


      [options] include:
         --make_benchmark : store the output of this run as the new
                            reference solution in the solver's tests/
                            sub-directory

         --compare_benchmark : compare the final result of this run
                               to the stored benchmark for this problem
                               (looking in the solver's tests/ sub-
                               directory).

      [runtime parameters] override any of the runtime defaults of
      parameters specified in the inputs file.  For instance, to turn
      off runtime visualization, add:

         vis.dovis=0

      to the end of the commandline.

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


# commandline argument defaults
makeBench = 0
compBench = 0

try: opts, next = getopt.getopt(sys.argv[1:], "", 
                                ["make_benchmark", "compare_benchmark"])

except getopt.GetoptError:
    msg.fail("invalid calling sequence")
    sys.exit(2)

for o, a in opts:

    if o == "--make_benchmark":
        makeBench = 1

    if o == "--compare_benchmark":
        compBench = 1

if makeBench and compBench:
    msg.fail("ERROR: cannot have both --make_benchmark and --compare_benchmark")


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


#-----------------------------------------------------------------------------
# benchmarks (for regression testing)
#-----------------------------------------------------------------------------
# are we comparing to a benchmark?
if compBench:
    compFile = solverName + "/tests/" + basename + "%4.4d" % (n)
    msg.warning("comparing to: %s " % (compFile) )
    benchGrid, benchData = patch.read(compFile)

    result = compare.compare(myGrid, myData, benchGrid, benchData)
    
    if result == 0:
        msg.success("results match benchmark\n")
    else:
        msg.fail("ERROR: " + compare.errors[result] + "\n")


# are we storing a benchmark?
if makeBench:
    benchFile = solverName + "/tests/" + basename + "%4.4d" % (n)
    msg.warning("storing new benchmark: %s\n " % (benchFile) )
    myData.write(benchFile)
    

#-----------------------------------------------------------------------------
# final reports
#-----------------------------------------------------------------------------
runparams.printUnusedParams()
profile.timeReport()

exec problemName + '.finalize()'
