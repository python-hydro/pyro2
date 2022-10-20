#!/usr/bin/env python3


import argparse
import datetime
import os
import sys

import examples.multigrid.mg_test_general_inhomogeneous as mg_test_general_inhomogeneous
import examples.multigrid.mg_test_simple as mg_test_simple
import examples.multigrid.mg_test_vc_dirichlet as mg_test_vc_dirichlet
import examples.multigrid.mg_test_vc_periodic as mg_test_vc_periodic
import pyro.pyro_sim as pyro


class PyroTest:
    def __init__(self, solver, problem, inputs, options):
        self.solver = solver
        self.problem = problem
        self.inputs = inputs
        self.options = options

    def __str__(self):
        return f"{self.solver}-{self.problem}"


def do_tests(out_file,
             reset_fails=False, store_all_benchmarks=False,
             single=None, solver=None, rtol=1e-12):

    opts = {"driver.verbose": 0, "vis.dovis": 0, "io.do_io": 0}

    results = {}

    tests = []
    tests.append(PyroTest("advection", "smooth", "inputs.smooth", opts))
    tests.append(PyroTest("advection_nonuniform",
                          "slotted", "inputs.slotted", opts))
    tests.append(PyroTest("advection_rk", "smooth", "inputs.smooth", opts))
    tests.append(PyroTest("advection_fv4",
                          "smooth", "inputs.smooth", opts))
    tests.append(PyroTest("compressible", "quad", "inputs.quad", opts))
    tests.append(PyroTest("compressible", "sod", "inputs.sod.x", opts))
    tests.append(PyroTest("compressible", "rt", "inputs.rt", opts))
    tests.append(PyroTest("compressible_rk", "rt", "inputs.rt", opts))
    tests.append(PyroTest("compressible_fv4", "acoustic_pulse",
                          "inputs.acoustic_pulse", opts))
    tests.append(PyroTest("compressible_sdc", "acoustic_pulse",
                          "inputs.acoustic_pulse", opts))
    tests.append(PyroTest("diffusion", "gaussian",
                          "inputs.gaussian", opts))
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
        p = pyro.PyroBenchmark(t.solver, comp_bench=True,
                               reset_bench_on_fail=reset_fails, make_bench=store_all_benchmarks)
        p.initialize_problem(t.problem, t.inputs, t.options)
        err = p.run_sim(rtol)

        results[str(t)] = err

    # standalone tests
    if single is None:
        bench_dir = os.path.dirname(os.path.realpath(__file__)) + "/multigrid/tests/"
        err = mg_test_simple.test_poisson_dirichlet(256, comp_bench=True, bench_dir=bench_dir,
                                                    store_bench=store_all_benchmarks, verbose=0)
        results["mg_poisson_dirichlet"] = err

        err = mg_test_vc_dirichlet.test_vc_poisson_dirichlet(512,
                                                             comp_bench=True, bench_dir=bench_dir,
                                                             store_bench=store_all_benchmarks, verbose=0)
        results["mg_vc_poisson_dirichlet"] = err

        err = mg_test_vc_periodic.test_vc_poisson_periodic(512, comp_bench=True, bench_dir=bench_dir,
                                                           store_bench=store_all_benchmarks,
                                                           verbose=0)
        results["mg_vc_poisson_periodic"] = err

        err = mg_test_general_inhomogeneous.test_general_poisson_inhomogeneous(512,
                                                                               comp_bench=True,
                                                                               bench_dir=bench_dir,
                                                                               store_bench=store_all_benchmarks,
                                                                               verbose=0)
        results["mg_general_poisson_inhomogeneous"] = err

    failed = 0

    out = [sys.stdout]
    if out_file is not None:
        out.append(open(out_file, "w"))

    for f in out:
        f.write("pyro tests run: {}\n\n".format(
            str(datetime.datetime.now().replace(microsecond=0))))

        for s, r in sorted(results.items()):
            if not r == 0:
                f.write(f"{s:42} failed\n")
                failed += 1
            else:
                f.write(f"{s:42} passed\n")

        f.write(f"\n{failed} test(s) failed\n")

        if not f == sys.stdout:
            f.close()

    return failed


if __name__ == "__main__":

    p = argparse.ArgumentParser()

    p.add_argument("-o",
                   help="name of file to output the report to (otherwise output to the screen",
                   type=str, nargs=1)

    p.add_argument("--single",
                   help="name of a single test (solver-problem) to run",
                   type=str, default=None)

    p.add_argument("--solver",
                   help="only test the solver specified",
                   type=str, default=None)

    p.add_argument("--reset_failures", "-r",
                   help="if a test fails, reset the benchmark",
                   action="store_true")

    p.add_argument("--store_all_benchmarks",
                   help="rewrite all the benchmarks, regardless of pass / fail",
                   action="store_true")

    p.add_argument("--rtol",
                   help="relative tolerance to use when comparing data to benchmarks",
                   type=float, nargs=1)

    args = p.parse_args()

    try:
        outfile = args.o[0]
    except TypeError:
        outfile = None

    try:
        rtol = args.rtol[0]
    except TypeError:
        rtol = 1.e-12

    failed = do_tests(outfile,
                      reset_fails=args.reset_failures,
                      store_all_benchmarks=args.store_all_benchmarks,
                      single=args.single, solver=args.solver, rtol=rtol)

    sys.exit(failed)
