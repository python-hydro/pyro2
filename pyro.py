#!/usr/bin/env python

import getopt
import os
import sys

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

tc = profile.TimerCollection()

tm_main = tc.timer("main")
tm_main.begin()

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
make_bench = 0
comp_bench = 0

try: opts, next = getopt.getopt(sys.argv[1:], "", 
                                ["make_benchmark", "compare_benchmark"])

except getopt.GetoptError:
    msg.fail("invalid calling sequence")
    sys.exit(2)

for o, a in opts:

    if o == "--make_benchmark":
        make_bench = 1

    if o == "--compare_benchmark":
        comp_bench = 1

if make_bench and comp_bench:
    msg.fail("ERROR: cannot have both --make_benchmark and --compare_benchmark")


try: solverName = next[0]
except IndexError:
    print usage
    msg.fail("ERROR: solver name not specified on command line")

try: problem_name = next[1]
except IndexError:
    print usage
    msg.fail("ERROR: problem name not specified on command line")

try: paramFile = next[2]
except IndexError:
    print usage
    msg.fail("ERROR: parameter file not specified on command line")


other_commands = []
if len(next) > 3:
    other_commands = next[3:]

# actually import the solver-specific stuff under the 'solver' namespace
exec 'import ' + solverName + ' as solver'


#-----------------------------------------------------------------------------
# runtime parameters
#-----------------------------------------------------------------------------

# parameter defaults
rp = runparams.RuntimeParameters()
rp.load_params("_defaults")
rp.load_params(solverName + "/_defaults")

# problem-specific runtime parameters
rp.load_params(solverName + "/problems/_" + problem_name + ".defaults")

# now read in the inputs file
if not os.path.isfile(paramFile):
    # check if the param file lives in the solver's problems directory
    paramFile = solverName + "/problems/" + paramFile
    if not os.path.isfile(paramFile):
        msg.fail("ERROR: inputs file does not exist")

rp.load_params(paramFile, no_new=1)

# and any commandline overrides
rp.command_line_params(other_commands)

# write out the inputs.auto
rp.print_paramfile()


#-----------------------------------------------------------------------------
# initialization
#-----------------------------------------------------------------------------

# initialize the Simulation object -- this will hold the grid and data and
# know about the runtime parameters and which problem we are running
sim = solver.Simulation(problem_name, rp, timers=tc)

sim.initialize()
sim.preevolve()


#-----------------------------------------------------------------------------
# evolve
#-----------------------------------------------------------------------------
tmax = rp.get_param("driver.tmax")
max_steps = rp.get_param("driver.max_steps")

init_tstep_factor = rp.get_param("driver.init_tstep_factor")
max_dt_change = rp.get_param("driver.max_dt_change")
fix_dt = rp.get_param("driver.fix_dt")

pylab.ion()

n = 0
sim.cc_data.t = 0.0

# output the 0th data
basename = rp.get_param("io.basename")
sim.cc_data.write(basename + "%4.4d" % (n))

dovis = rp.get_param("vis.dovis")
if dovis: 
    pylab.figure(num=1, figsize=(8,6), dpi=100, facecolor='w')
    sim.dovis(0)
    
nout = 0

while sim.cc_data.t < tmax and n < max_steps:

    # fill boundary conditions
    tm_bc = tc.timer("fill_bc")
    tm_bc.begin()
    sim.cc_data.fill_BC_all()
    tm_bc.end()

    # get the timestep
    dt = sim.timestep()
    if fix_dt > 0.0:
        dt = fix_dt
    else:
        if n == 0:
            dt = init_tstep_factor*dt
            dt_old = dt
        else:
            dt = min(max_dt_change*dt_old, dt)
            dt_old = dt
            
    if sim.cc_data.t + dt > tmax:
        dt = tmax - sim.cc_data.t

    # evolve for a single timestep
    sim.evolve(dt)


    # increment the time
    sim.cc_data.t += dt
    n += 1
    print "%5d %10.5f %10.5f" % (n, sim.cc_data.t, dt)


    # output
    dt_out = rp.get_param("io.dt_out")
    n_out = rp.get_param("io.n_out")

    if sim.cc_data.t >= (nout + 1)*dt_out or n%n_out == 0:

        tm_io = tc.timer("output")
        tm_io.begin()

        msg.warning("outputting...")
        basename = rp.get_param("io.basename")
        sim.cc_data.write(basename + "%4.4d" % (n))
        nout += 1

        tm_io.end()


    # visualization
    if dovis: 
        tm_vis = tc.timer("vis")
        tm_vis.begin()

        sim.dovis(n)
        store = rp.get_param("vis.store_images")

        if store == 1:
            basename = rp.get_param("io.basename")
            pylab.savefig(basename + "%4.4d" % (n) + ".png")

        tm_vis.end()

tm_main.end()


#-----------------------------------------------------------------------------
# benchmarks (for regression testing)
#-----------------------------------------------------------------------------
# are we comparing to a benchmark?
if comp_bench:
    compare_file = solverName + "/tests/" + basename + "%4.4d" % (n)
    msg.warning("comparing to: %s " % (compare_file) )
    bench_grid, bench_data = patch.read(compare_file)

    result = compare.compare(sim.cc_data.grid, sim.cc_data, bench_grid, bench_data)
    
    if result == 0:
        msg.success("results match benchmark\n")
    else:
        msg.fail("ERROR: " + compare.errors[result] + "\n")


# are we storing a benchmark?
if make_bench:
    bench_file = solverName + "/tests/" + basename + "%4.4d" % (n)
    msg.warning("storing new benchmark: %s\n " % (bench_file) )
    sim.cc_data.write(bench_file)
    

#-----------------------------------------------------------------------------
# final reports
#-----------------------------------------------------------------------------
rp.print_unused_params()
tc.report()

sim.finalize()

