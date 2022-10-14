#!/usr/bin/env python3


import importlib
import sys

import numpy as np

import pyro.mesh.patch as patch
import pyro.util.io_pyro as io

usage = """
      compare the output in file from the smooth advection problem to
      the analytic solution.

      usage: ./smooth_error.py file
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

# create a new data object on the same grid
analytic = patch.CellCenterData2d(myg, dtype=np.float64)

bco = myd.BCs[myd.names[0]]
analytic.register_var("density", bco)
analytic.create()

# use the original initialization routine to set the analytic solution
problem = importlib.import_module(f"pyro.{sim.solver_name}.problems.{sim.problem_name}")
problem.init_data(analytic, None)

# compare the error
dens_numerical = myd.get_var("density")
dens_analytic = analytic.get_var("density")

print("mesh details")
print(myg)

e = dens_numerical - dens_analytic

print("error norms (absolute, relative): ", e.norm(), e.norm()/dens_analytic.norm())
