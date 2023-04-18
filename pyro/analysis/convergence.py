#!/usr/bin/env python3

import sys

import numpy as np

import pyro.util.io_pyro as io

# read in two files -- one twice the resolution of the other, and
# compute the error by averaging down

usage = """
      usage: ./convergence.py fine coarse variable_name[optional, default=density]
"""


def compare(fine, coarse, var):

    dens = coarse.get_var(var)
    dens_avg = fine.restrict(var, N=2)

    e = coarse.grid.scratch_array()
    e.v()[:, :] = dens.v() - dens_avg.v()

    return float(np.abs(e).max()), e.norm()


def main():
    if (len(sys.argv) > 4 or len(sys.argv) < 3):
        print(usage)
        sys.exit(2)

    fine = sys.argv[1]
    coarse = sys.argv[2]

    var = "density"
    if len(sys.argv) == 4:
        var = sys.argv[3]

    ff = io.read(fine)
    cc = io.read(coarse)

    result = compare(ff.cc_data, cc.cc_data, var)

    print(f"inf norm and L2 norm of {var}: ", result)


if __name__ == "__main__":
    main()
