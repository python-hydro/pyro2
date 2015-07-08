#!/usr/bin/env python

from __future__ import print_function

import argparse
import datetime
import os
import sys

import pyro
import multigrid.test_mg as test_mg
import multigrid.test_mg_vc_dirichlet as test_mg_vc_dirichlet
import multigrid.test_mg_vc_periodic as test_mg_vc_periodic
import multigrid.test_mg_general_inhomogeneous as test_mg_general_inhomogeneous

class test(object):
    def __init__(self, solver, problem, inputs, options):
        self.solver = solver
        self.problem = problem
        self.inputs = inputs
        self.options = options


def do_tests(build, out_file, do_standalone=True):

    # make sure we've built stuff
    print("build = ", build)

    if build: os.system("./mk.sh")
    
    opts = "driver.verbose=0 vis.dovis=0 io.do_io=0".split()

    tests = []
    tests.append(test("advection", "smooth", "inputs.smooth", opts))
    tests.append(test("compressible", "quad", "inputs.quad", opts))
    tests.append(test("compressible", "sod", "inputs.sod.x", opts))
    tests.append(test("compressible", "rt", "inputs.rt", opts))
    tests.append(test("diffusion", "gaussian", "inputs.gaussian", opts))
    tests.append(test("incompressible", "shear", "inputs.shear", opts))
    tests.append(test("lm_atm", "bubble", "inputs.bubble", opts))


    results = {}

    for t in tests:
        err = pyro.doit(t.solver, t.problem, t.inputs,
                        other_commands=t.options, comp_bench=True)
        results["{}-{}".format(t.solver, t.problem)] = err


    # standalone tests
    if do_standalone:
        err = test_mg.test_poisson_dirichlet(256, comp_bench=True, verbose=0)
        results["mg_poisson_dirichlet"] = err

        err = test_mg_vc_dirichlet.test_vc_poisson_dirichlet(512, comp_bench=True, verbose=0)
        results["mg_vc_poisson_dirichlet"] = err

        err = test_mg_vc_periodic.test_vc_poisson_periodic(512, comp_bench=True, verbose=0)
        results["mg_vc_poisson_periodic"] = err

        err = test_mg_general_inhomogeneous.test_general_poisson_inhomogeneous(512, comp_bench=True, verbose=0)
        results["mg_general_poisson_inhomogeneous"] = err    


    failed = 0

    out = [sys.stdout]
    if not out_file == None:
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

        if not f == sys.stdout: f.close()
    

if __name__ == "__main__":

    p = argparse.ArgumentParser()

    p.add_argument("-o",
                   help="name of file to output the report to (otherwise output to the screen",
                   type=str, nargs=1)

    p.add_argument("--build",
                   help="execute the mk.sh script first before any tests",
                   action="store_true")
    
    args = p.parse_args()

    try: outfile = args.o[0]
    except: outfile = None

    build = args.build
    
    do_tests(build, outfile)


        
