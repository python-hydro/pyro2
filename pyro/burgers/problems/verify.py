#!/usr/bin/env python3

import sys

import numpy as np

import pyro.util.io_pyro as io

usage = """
usage: ./verify.py file1 file2
Used to verify the shock speed for inviscid Burgers' solver.
Input files should be input files from burgers/problem/test.py
"""


def verify(file1, file2):

    s1 = io.read(file1)
    s2 = io.read(file2)

    d1 = s1.cc_data
    d2 = s2.cc_data

    dt = abs(d1.t - d2.t)

    u1 = d1.get_var("x-velocity")
    u2 = d2.get_var("x-velocity")

    dis_index1 = np.where(u1 == np.max(u1))
    dis_index2 = np.where(u2 == np.max(u2))

    dx = np.max(d1.grid.x2d[dis_index1]) - np.max(d2.grid.x2d[dis_index2])
    
    shock_speed_theo = 1.5
    shock_speed_sim = dx / dt

    print(f"Theoretical shock speed is: {shock_speed_theo}")
    print(f"Shock speed from simulation is: {shock_speed_sim}")

    if np.isclose(shock_speed_sim, shock_speed_theo):
        print("SUCCESS, shock speeds match")
    else:
        print("ERROR, shock speeds don't match") 

if __name__ == "__main__":

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    verify(file1, file2)
