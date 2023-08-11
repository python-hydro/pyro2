#!/usr/bin/env python3


import argparse
import contextlib
import datetime
import io
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import pyro.pyro_sim as pyro
from pyro.multigrid.examples import (mg_test_general_inhomogeneous,
                                     mg_test_simple, mg_test_vc_dirichlet,
                                     mg_test_vc_periodic)


class PyroTest:
    def __init__(self, solver, problem, inputs, options):
        self.solver = solver
        self.problem = problem
        self.inputs = inputs
        self.options = options

    def __str__(self):
        return f"{self.solver}-{self.problem}"


@contextlib.contextmanager
def avoid_interleaved_output(nproc):
    """Collect all the printed output and print it all at once to avoid interleaving."""
    if nproc == 1:
        # not running in parallel, so we don't have to worry about interleaving
        yield
    else:
        output_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(output_buffer), \
                 contextlib.redirect_stderr(output_buffer):
                yield
        finally:
            # a single print call probably won't get interleaved
            print(output_buffer.getvalue(), end="", flush=True)


def run_test(t, reset_fails, store_all_benchmarks, rtol, nproc):
    orig_cwd = Path.cwd()
    # run each test in its own directory, since some of the output file names
    # overlap between tests, and h5py needs exclusive access when writing
    test_dir = orig_cwd / f"test_outputs/{t}"
    test_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(test_dir)
        with avoid_interleaved_output(nproc):
            p = pyro.PyroBenchmark(t.solver, comp_bench=True,
                                   reset_bench_on_fail=reset_fails,
                                   make_bench=store_all_benchmarks)
            p.initialize_problem(t.problem, t.inputs, t.options)
            start_n = p.sim.n
            err = p.run_sim(rtol)
    finally:
        os.chdir(orig_cwd)
    if err == 0:
        # the test passed; clean up the output files for developer use
        basename = p.rp.get_param("io.basename")
        (test_dir / f"{basename}{start_n:04d}.h5").unlink()
        (test_dir / f"{basename}{p.sim.n:04d}.h5").unlink()
        (test_dir / "inputs.auto").unlink()
        test_dir.rmdir()
        # try removing the top-level output directory
        try:
            test_dir.parent.rmdir()
        except OSError:
            pass

    return str(t), err


def run_test_star(args):
    """multiprocessing doesn't like lambdas, so this needs to be a full function"""
    return run_test(*args)


def do_tests(out_file,
             reset_fails=False, store_all_benchmarks=False,
             single=None, solver=None, rtol=1e-12, nproc=1):

    opts = {"driver.verbose": 0, "vis.dovis": 0, "io.do_io": 0}

    results = {}

    tests = []
    tests.append(PyroTest("advection", "smooth", "inputs.smooth", opts))
    tests.append(PyroTest("advection_ppm",
                          "smooth", "inputs.smooth", opts))
    tests.append(PyroTest("advection_nonuniform",
                          "slotted", "inputs.slotted", opts))
    tests.append(PyroTest("advection_rk", "smooth", "inputs.smooth", opts))
    tests.append(PyroTest("advection_fv4",
                          "smooth", "inputs.smooth", opts))
    tests.append(PyroTest("burgers", "test", "inputs.test", opts))
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
    tests.append(PyroTest("incompressible_viscous", "cavity", "inputs.cavity", opts))
    tests.append(PyroTest("lm_atm", "bubble", "inputs.bubble", opts))
    tests.append(PyroTest("swe", "dam", "inputs.dam.x", opts))

    if single is not None:
        tests_to_run = [q for q in tests if str(q) == single]
    elif solver is not None:
        tests_to_run = [q for q in tests if q.solver == solver]
    else:
        tests_to_run = tests

    if nproc == 0:
        nproc = os.cpu_count()
    # don't create more processes than needed
    nproc = min(nproc, len(tests_to_run))
    with Pool(processes=nproc) as pool:
        tasks = ((t, reset_fails, store_all_benchmarks, rtol, nproc) for t in tests_to_run)
        imap_it = pool.imap_unordered(run_test_star, tasks)
        # collect run results
        for name, err in imap_it:
            results[name] = err

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

        if f != sys.stdout:
            f.close()

    return failed


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--outfile", "-o",
                   help="name of file to output the report to (in addition to the screen)",
                   type=str, default=None)

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
                   type=float, default=1.e-12)

    p.add_argument("--nproc", "-n",
                   help="maximum number of parallel processes to run, or 0 to use all cores",
                   type=int, default=1)

    args = p.parse_args()

    failed = do_tests(args.outfile,
                      reset_fails=args.reset_failures,
                      store_all_benchmarks=args.store_all_benchmarks,
                      single=args.single, solver=args.solver, rtol=args.rtol,
                      nproc=args.nproc)

    sys.exit(failed)


if __name__ == "__main__":
    main()
