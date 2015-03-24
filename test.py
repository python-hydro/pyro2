#!/usr/bin/env python

from __future__ import print_function

import argparse
import datetime
import sys

import pyro
import multigrid.test_mg as test_mg
import multigrid.test_mg_vc_dirichlet as test_mg_vc_dirichlet
import multigrid.test_mg_vc_periodic as test_mg_vc_periodic

class test:
    def __init__(self, solver, problem, inputs, options):
        self.solver = solver
        self.problem = problem
        self.inputs = inputs
        self.options = options


def do_tests(out_file):
    
    opts = "driver.verbose=0 vis.dovis=0 io.do_io=0".split()

    tests = []
    tests.append(test("advection", "smooth", "inputs.smooth", opts))
    tests.append(test("compressible", "quad", "inputs.quad", opts))
    tests.append(test("diffusion", "gaussian", "inputs.gaussian", opts))
    tests.append(test("incompressible", "shear", "inputs.shear", opts))
    tests.append(test("lm_atm", "bubble", "inputs.bubble", opts))


    results = {}

    for t in tests:
        err = pyro.doit(t.solver, t.problem, t.inputs,
                        other_commands=t.options, comp_bench=True)
        results[t.solver] = err


    # standalone tests
    err = test_mg.test_poisson_dirichlet(256, comp_bench=True)
    results["mg_poisson_dirichlet"] = err

    err = test_mg_vc_dirichlet.test_vc_poisson_dirichlet(512, comp_bench=True)
    results["mg_vc_poisson_dirichlet"] = err

    err = test_mg_vc_periodic.test_vc_poisson_periodic(512, comp_bench=True)
    results["mg_vc_poisson_periodic"] = err


    failed = 0

    f = open(out_file, "w") if out_file else sys.stdout

    f.write("pyro tests run: {}\n\n".format(str(datetime.datetime.now().replace(microsecond=0))))
    
    for s, r in sorted(results.items()):
        if not r == 0:
            f.write("{:32} failed\n".format(s))
            failed += 1
        else:
            f.write("{:32} passed\n".format(s))


    f.write("\n{} test(s) failed\n".format(failed))

    if out_file: f.close()
    

if __name__ == "__main__":

    p = argparse.ArgumentParser()

    p.add_argument("-o",
                   help="name of file to output the report to (otherwise output to the screen",
                   type=str, nargs=1)

    args = p.parse_args()

    do_tests(args.o[0])


        
