#!/usr/bin/env python3

from __future__ import print_function

import argparse
import datetime
import os
import sys

import pytest

import pyro
import examples.multigrid.mg_test_simple as mg_test_simple
import examples.multigrid.mg_test_vc_dirichlet as mg_test_vc_dirichlet
import examples.multigrid.mg_test_vc_periodic as mg_test_vc_periodic
import examples.multigrid.mg_test_general_inhomogeneous as mg_test_general_inhomogeneous


class PyroTest(object):
    def __init__(self, solver, problem, inputs, options):
        self.solver = solver
        self.problem = problem
        self.inputs = inputs
        self.options = options

    def __str__(self):
        return "{}-{}".format(self.solver, self.problem)


def do_tests(build, out_file, do_standalone=True, do_main=True,
             reset_fails=False, store_all_benchmarks=False,
             single=None, solver=None):

    # make sure we've built stuff
    print("build = ", build)

    if build:
        os.system("./mk.sh")

    opts = "driver.verbose=0 vis.dovis=0 io.do_io=0".split()

    results = {}

    if do_main:
        tests = []
        tests.append(PyroTest("advection", "smooth", "inputs.smooth", opts))
        tests.append(PyroTest("advection_rk", "smooth", "inputs.smooth", opts))
        tests.append(PyroTest("advection_fv4", "smooth", "inputs.smooth", opts))
        tests.append(PyroTest("compressible", "quad", "inputs.quad", opts))
        tests.append(PyroTest("compressible", "sod", "inputs.sod.x", opts))
        tests.append(PyroTest("compressible", "rt", "inputs.rt", opts))
        tests.append(PyroTest("compressible_rk", "rt", "inputs.rt", opts))
        tests.append(PyroTest("compressible_fv4", "acoustic_pulse", "inputs.acoustic_pulse", opts))
        tests.append(PyroTest("compressible_sdc", "acoustic_pulse", "inputs.acoustic_pulse", opts))
        tests.append(PyroTest("diffusion", "gaussian", "inputs.gaussian", opts))
        tests.append(PyroTest("incompressible", "shear", "inputs.shear", opts))
        tests.append(PyroTest("lm_atm", "bubble", "inputs.bubble", opts))
        tests.append(PyroTest("swe", "dam", "inputs.dam.x", opts))

        if single is not None:
            tests_to_run = [q for q in tests if str(q) == single]
        elif solver is not None:
            tests_to_run = [q for q in tests if q.solver == solver]
        else:
            tests_to_run = tests

        for t in tests_to_run:
            err = pyro.doit(t.solver, t.problem, t.inputs,
                            other_commands=t.options, comp_bench=True,
                            reset_bench_on_fail=reset_fails, make_bench=store_all_benchmarks)
            results[str(t)] = err

    # standalone tests
    if do_standalone and single is None:
        err = mg_test_simple.test_poisson_dirichlet(256, comp_bench=True,
                                                    store_bench=store_all_benchmarks, verbose=0)
        results["mg_poisson_dirichlet"] = err

        err = mg_test_vc_dirichlet.test_vc_poisson_dirichlet(512,
                                                             comp_bench=True,
                                                             store_bench=store_all_benchmarks, verbose=0)
        results["mg_vc_poisson_dirichlet"] = err

        err = mg_test_vc_periodic.test_vc_poisson_periodic(512, comp_bench=True,
                                                           store_bench=store_all_benchmarks,
                                                           verbose=0)
        results["mg_vc_poisson_periodic"] = err

        err = mg_test_general_inhomogeneous.test_general_poisson_inhomogeneous(512,
                                                                               comp_bench=True,
                                                                               store_bench=store_all_benchmarks,
                                                                               verbose=0)
        results["mg_general_poisson_inhomogeneous"] = err

    failed = 0

    out = [sys.stdout]
    if out_file is not None:
        out.append(open(out_file, "w"))

    for f in out:
        f.write("pyro tests run: {}\n\n".format(str(datetime.datetime.now().replace(microsecond=0))))

        for s, r in sorted(results.items()):
            if not r == 0:
                f.write("{:42} failed\n".format(s))
                failed += 1
            else:
                f.write("{:42} passed\n".format(s))

        f.write("\n{} test(s) failed\n".format(failed))

        if not f == sys.stdout:
            f.close()


if __name__ == "__main__":

    p = argparse.ArgumentParser()

    p.add_argument("-o",
                   help="name of file to output the report to (otherwise output to the screen",
                   type=str, nargs=1)

    p.add_argument("--build",
                   help="execute the mk.sh script first before any tests",
                   action="store_true")

    p.add_argument("--single",
                   help="name of a single test (solver-problem) to run",
                   type=str, default=None)

    p.add_argument("--solver",
                   help="only test the solver specified",
                   type=str, default=None)

    p.add_argument("--skip_standalone",
                   help="skip the tests that don't go through pyro.py",
                   action="store_true")

    p.add_argument("--reset_failures", "-r",
                   help="if a test fails, reset the benchmark",
                   action="store_true")

    p.add_argument("--store_all_benchmarks",
                   help="rewrite all the benchmarks, regardless of pass / fail",
                   action="store_true")

    p.add_argument("--skip_main",
                   help="skip the tests that go through pyro.py, and only run standalone tests",
                   action="store_true")

    p.add_argument("--unittests_only", "-u",
                   help="only do the unit tests",
                   action="store_true")

    args = p.parse_args()

    try:
        outfile = args.o[0]
    except TypeError:
        outfile = None

    build = args.build

    do_main = True
    if args.skip_main:
        do_main = False

    do_standalone = True
    if args.skip_standalone:
        do_standalone = False

    if not args.unittests_only:
        do_tests(build, outfile, do_standalone=do_standalone, do_main=do_main,
                 reset_fails=args.reset_failures,
                 store_all_benchmarks=args.store_all_benchmarks,
                 single=args.single, solver=args.solver)

    # unit tests
    if args.single is None:
        pytest.main(["-v"])
