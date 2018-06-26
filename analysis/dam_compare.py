#!/usr/bin/env python3

from __future__ import print_function

import numpy as np
from scipy.optimize import brentq
import sys
import os
import matplotlib.pyplot as plt
from util import msg, runparams, io

usage = """
      compare the output for a dam problem with the exact solution contained
      in dam-exact.out.

      usage: ./dam_compare.py file
"""


def abort(string):
    print(string)
    sys.exit(2)


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

# time of file
t = myd.t
if myg.nx > myg.ny:
    # x-problem
    xmin = myg.xmin
    xmax = myg.xmax
    param_file = "inputs.dam.x"
else:
    # y-problem
    xmin = myg.ymin
    xmax = myg.ymax
    param_file = "inputs.dam.y"


height = myd.get_var("height")
xmom = myd.get_var("x-momentum")
ymom = myd.get_var("y-momentum")

# get the 1-d profile from the simulation data -- assume that whichever
# coordinate is the longer one is the direction of the problem

# parameter defaults
rp = runparams.RuntimeParameters()
rp.load_params("../_defaults")
rp.load_params("../swe/_defaults")
rp.load_params("../swe/problems/_dam.defaults")

# now read in the inputs file
if not os.path.isfile(param_file):
    # check if the param file lives in the solver's problems directory
    param_file = "../swe/problems/" + param_file
    if not os.path.isfile(param_file):
        msg.fail("ERROR: inputs file does not exist")

rp.load_params(param_file, no_new=1)

if myg.nx > myg.ny:
    # x-problem
    x = myg.x[myg.ilo:myg.ihi+1]
    jj = myg.ny//2

    h = height[myg.ilo:myg.ihi+1, jj]

    u = xmom[myg.ilo:myg.ihi+1, jj]/h
    ut = ymom[myg.ilo:myg.ihi+1, jj]/h

else:

    # y-problem
    x = myg.y[myg.jlo:myg.jhi+1]
    ii = myg.nx//2

    h = height[ii, myg.jlo:myg.jhi+1]

    u = ymom[ii, myg.jlo:myg.jhi+1]/h
    ut = xmom[ii, myg.jlo:myg.jhi+1]/h

print(myg)

x_exact = x
h_exact = np.zeros_like(x)
u_exact = np.zeros_like(x)

# find h0, h1
h1 = rp.get_param("dam.h_left")
h0 = rp.get_param("dam.h_right")


def find_h2(h2):
    return (h2/h1)**3 - 9*(h2/h1)**2*(h0/h1) + \
        16*(h2/h1)**1.5*(h0/h1) - (h2/h1)*(h0/h1)*(h0/h1+8) + \
        (h0/h1)**3


h2 = brentq(find_h2, min(h0, h1), max(h0, h1))

# calculate sound speeds
g = rp.get_param("swe.grav")

c0 = np.sqrt(g*h0)
c1 = np.sqrt(g*h1)
c2 = np.sqrt(g*h2)

u2 = 2 * (c1 - c2)

# shock speed
xi = c0 * np.sqrt(1/8 * ((2*(c2/c0)**2 + 1)**2 - 1))

xctr = 0.5*(xmin + xmax)

# h0
idx = x >= xctr + xi*t
h_exact[idx] = h0
u_exact[idx] = 0

# h1
idx = x <= xctr - c1*t
h_exact[idx] = h1
u_exact[idx] = 0

# h2
idx = ((x >= xctr + (u2-c2)*t) & (x < xctr + xi*t))
h_exact[idx] = h2
u_exact[idx] = u2

# h3
idx = ((x >= xctr - c1*t) & (x < xctr + (u2-c2)*t))
c3 = 1/3 * (2*c1 - (x-xctr)/t)
h_exact[idx] = c3[idx]**2 / g
u_exact[idx] = 2 * (c1-c3[idx])

# plot
fig, axes = plt.subplots(nrows=2, ncols=1, num=1)

plt.rc("font", size=10)

ax = axes.flat[0]

ax.plot(x_exact, h_exact, label='Exact')
ax.scatter(x, h, marker="x", s=7, color="r", label='Pyro')

ax.set_ylabel(r"$h$")
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.1)

ax = axes.flat[1]

ax.plot(x_exact, u_exact)
ax.scatter(x, u, marker="x", s=7, color="r")

ax.set_ylabel(r"$u$")
ax.set_xlim(0, 1.0)

if (myg.nx > myg.ny):
    ax.set_xlabel(r"x")
else:
    ax.set_xlabel(r"y")

lgd = axes.flat[0].legend()

plt.subplots_adjust(hspace=0.25)

fig.set_size_inches(8.0, 8.0)

plt.savefig("dam_compare.png", bbox_inches="tight")
