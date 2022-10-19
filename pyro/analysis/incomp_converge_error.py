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


if not len(sys.argv) == 2:
    print(usage)
    sys.exit(2)


try:
    file1 = sys.argv[1]
except IndexError:
    print(usage)
    sys.exit(2)

sim = io.read(file1)
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

print("errors: ", udiff.norm(), vdiff.norm())
