#!/usr/bin/env python3


import math
import sys

import numpy as np

import pyro.util.io_pyro as io

usage = """
      compare the output in file from the incompressible converge problem to
      the analytic solution.

      usage: ./incomp_converge_error.py file
"""


def get_errors(file):

    sim = io.read(file)
    myd = sim.cc_data
    myg = myd.grid

    # numerical solution
    u = myd.get_var("x-velocity")
    v = myd.get_var("y-velocity")

    t = myd.t

    # analytic solution
    u_exact = myg.scratch_array()
    u_exact[:, :] = 1.0 - 2.0*np.cos(2.0*math.pi*(myg.x2d-t))*np.sin(2.0*math.pi*(myg.y2d-t))

    v_exact = myg.scratch_array()
    v_exact[:, :] = 1.0 + 2.0*np.sin(2.0*math.pi*(myg.x2d-t))*np.cos(2.0*math.pi*(myg.y2d-t))

    # error
    udiff = u_exact - u
    vdiff = v_exact - v

    return udiff.norm(), vdiff.norm()


def main():
    if len(sys.argv) != 2:
        print(usage)
        sys.exit(2)

    file = sys.argv[1]
    errors = get_errors(file)
    print("errors: ", errors[0], errors[1])


if __name__ == "__main__":
    main()
