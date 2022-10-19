#!/usr/bin/env python3

import sys

import numpy as np

import pyro.util.io_pyro as io

# read in two files -- one twice the resolution of the other, and
# compute the error by averaging down

usage = """
      usage: ./convergence.py fine coarse
"""


def compare(fine, coarse):

    dens = coarse.get_var("density")
    dens_avg = fine.restrict("density", N=2)

    e = coarse.grid.scratch_array()
    e.v()[:, :] = dens.v() - dens_avg.v()

    return float(np.abs(e).max()), e.norm()


if __name__ == "__main__":

    if not len(sys.argv) == 3:
        print(usage)
        sys.exit(2)

    fine = sys.argv[1]
    coarse = sys.argv[2]

    ff = io.read(fine)
    cc = io.read(coarse)

    result = compare(ff.cc_data, cc.cc_data)

    print("inf/L2 norm of density: ", result)
