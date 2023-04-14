#!/usr/bin/env python3

import sys

import numpy as np

import pyro.util.io_pyro as io

"""
usage: ./verify.py file1 file2
Used to verify the shock speed for inviscid Burgers' solver.
Input files should be input files from burgers/problem/test.py
"""


def verify(file1, file2):

    s1 = io.read(file1)
    s2 = io.read(file2)

    d1 = s1.cc_data
    d2 = s2.cc_data

    dt = d2.t - d1.t

    u1 = d1.get_var("x-velocity")
    u2 = d2.get_var("x-velocity")

    v1 = d1.get_var("y-velocity")
    v2 = d2.get_var("y-velocity")

    grid = d1.grid.x[d1.grid.ilo:d1.grid.ihi]

    # Do diagonal averaging:
    nx = len(grid)

    uv1_averages = []
    uv2_averages = []

    uv1 = np.sqrt(u1.v()*u1.v() + v1.v()*v1.v())
    uv2 = np.sqrt(u2.v()*u2.v() + v2.v()*v2.v())

    for n in range(-(nx-1), nx):

        diag_uv1 = np.diagonal(np.flipud(uv1), n)
        uv1_averages.append(diag_uv1.mean())

        diag_uv2 = np.diagonal(np.flipud(uv2), n)
        uv2_averages.append(diag_uv2.mean())

    shock_speed_theo = np.sqrt(2.0*2.0 + 2.0*2.0)

    x = [grid[0]]

    for n in grid[1:]:

        x.append(0.5 * (x[-1] + n))
        x.append(n)

    # When the speed first drops to 0.9S

    disconX1 = [x[n] for n, uv in enumerate(uv1_averages) if (uv < 0.9*shock_speed_theo and x[n] > 0.5)]
    disconX2 = [x[n] for n, uv in enumerate(uv2_averages) if (uv < 0.9*shock_speed_theo and x[n] > 0.5)]

    dx = disconX1[0] - disconX2[0]
    shock_speed_sim = np.sqrt(2.0 * (dx/dt) * (dx/dt))

    print(f"Theoretical shock speed is: {shock_speed_theo}")
    print(f"Shock speed from simulation is: {shock_speed_sim}")

    if np.isclose(shock_speed_sim, shock_speed_theo):
        print("SUCCESS, shock speeds match")
    else:
        print("ERROR, shock speeds don't match")


if __name__ == "__main__":

    file1_ = sys.argv[1]
    file2_ = sys.argv[2]

    verify(file1_, file2_)
