#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import sys

import matplotlib.pyplot as plt

import compare
import mesh.patch as patch
from util import msg, profile, runparams


def doit(solver_name, problem_name, param_file,
         other_commands=None,
         comp_bench=False, make_bench=False):

    msg.bold('pyro ...')

    tc = profile.TimerCollection()

    tm_main = tc.timer("main")
    tm_main.begin()

    #-------------------------------------------------------------------------
    # solver setup
    #-------------------------------------------------------------------------

    # actually import the solver-specific stuff under the 'solver' namespace
    exec('import ' + solver_name + ' as solver')


    #-------------------------------------------------------------------------
    # runtime parameters
    #-------------------------------------------------------------------------

    # parameter defaults
    rp = runparams.RuntimeParameters()
    rp.load_params("_defaults")
    rp.load_params(solver_name + "/_defaults")

    # problem-specific runtime parameters
    rp.load_params(solver_name + "/problems/_" + problem_name + ".defaults")

    # now read in the inputs file
    if not os.path.isfile(param_file):
        # check if the param file lives in the solver's problems directory
        param_file = solver_name + "/problems/" + param_file
        if not os.path.isfile(param_file):
            msg.fail("ERROR: inputs file does not exist")

    rp.load_params(param_file, no_new=1)

    # and any commandline overrides
    if not other_commands == None:
        rp.command_line_params(other_commands)

    # write out the inputs.auto
    rp.print_paramfile()


    #-------------------------------------------------------------------------
    # initialization
    #-------------------------------------------------------------------------

    # initialize the Simulation object -- this will hold the grid and
    # data and know about the runtime parameters and which problem we
    # are running
    sim = solver.Simulation(problem_name, rp, timers=tc)

    sim.initialize()
    sim.preevolve()


    #-------------------------------------------------------------------------
    # evolve
    #-------------------------------------------------------------------------
    tmax = rp.get_param("driver.tmax")
    max_steps = rp.get_param("driver.max_steps")

    init_tstep_factor = rp.get_param("driver.init_tstep_factor")
    max_dt_change = rp.get_param("driver.max_dt_change")
    fix_dt = rp.get_param("driver.fix_dt")

    verbose = rp.get_param("driver.verbose")

    plt.ion()

    n = 0
    sim.cc_data.t = 0.0

    # output the 0th data
    basename = rp.get_param("io.basename")
    sim.cc_data.write(basename + "%4.4d" % (n))

    dovis = rp.get_param("vis.dovis")
    if dovis:
        plt.figure(num=1, figsize=(8,6), dpi=100, facecolor='w')
        sim.dovis()

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
        if verbose > 0: print("%5d %10.5f %10.5f" % (n, sim.cc_data.t, dt))


        # output
        dt_out = rp.get_param("io.dt_out")
        n_out = rp.get_param("io.n_out")

        if sim.cc_data.t >= (nout + 1)*dt_out or n%n_out == 0:

            tm_io = tc.timer("output")
            tm_io.begin()

            if verbose > 0: msg.warning("outputting...")
            basename = rp.get_param("io.basename")
            sim.cc_data.write(basename + "%4.4d" % (n))
            nout += 1

            tm_io.end()


        # visualization
        if dovis:
            tm_vis = tc.timer("vis")
            tm_vis.begin()

            sim.dovis()
            store = rp.get_param("vis.store_images")

            if store == 1:
                basename = rp.get_param("io.basename")
                plt.savefig(basename + "%4.4d" % (n) + ".png")

            tm_vis.end()

    tm_main.end()


    #-------------------------------------------------------------------------
    # benchmarks (for regression testing)
    #-------------------------------------------------------------------------
    # are we comparing to a benchmark?
    if comp_bench:
        compare_file = solver_name + "/tests/" + basename + "%4.4d" % (n)
        msg.warning("comparing to: %s " % (compare_file) )
        bench_grid, bench_data = patch.read(compare_file)

        result = compare.compare(sim.cc_data.grid, sim.cc_data, bench_grid, bench_data)

        if result == 0:
            msg.success("results match benchmark\n")
        else:
            msg.fail("ERROR: " + compare.errors[result] + "\n")


    # are we storing a benchmark?
    if make_bench:
        if not os.path.isdir(solver_name + "/tests/"):
            try: os.mkdir(solver_name + "/tests/")
            except:
                msg.fail("ERROR: unable to create the solver's tests/ directory")
            
        bench_file = solver_name + "/tests/" + basename + "%4.4d" % (n)
        msg.warning("storing new benchmark: %s\n " % (bench_file) )
        sim.cc_data.write(bench_file)


    #-------------------------------------------------------------------------
    # final reports
    #-------------------------------------------------------------------------
    if verbose > 0: rp.print_unused_params()
    if verbose > 0: tc.report()

    sim.finalize()

    if comp_bench:
        return result
    else:
        return None
    

if __name__ == "__main__":

    p = argparse.ArgumentParser()

    p.add_argument("--make_benchmark",
                                  help="create a new benchmark file for regression testing",
                                  action="store_true")
    p.add_argument("--compare_benchmark",
                                  help="compare the end result to the stored benchmark",
                                  action="store_true")

    p.add_argument("solver", metavar="solver-name", type=str, nargs=1,
                                  help="name of the solver to use")
    p.add_argument("problem", metavar="problem-name", type=str, nargs=1,
                                  help="name of the problem to run")
    p.add_argument("param", metavar="inputs-file", type=str, nargs=1,
                                  help="name of the inputs file")

    p.add_argument("other", metavar="runtime-parameters", type=str, nargs="*",
                                  help="additional runtime parameters that override the inputs file")

    args = p.parse_args()


    doit(args.solver[0], args.problem[0], args.param[0],
         other_commands=args.other,
         comp_bench=args.compare_benchmark,
         make_bench=args.make_benchmark)

