#!/usr/bin/env python3


import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import pyro.util.io_pyro as io

mpl.rcParams["text.usetex"] = True
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

# font sizes
mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'

usage = """
      compare the output for a Sedov problem with the exact solution contained
      in cylindrical-sedov.out.  To do this, we need to bin the Sedov data
      into radial bins."""

parser = argparse.ArgumentParser(description=usage)
parser.add_argument("-o", type=str, default="sedov_compare.png",
                    metavar="plot.png", help="output file name")
parser.add_argument("plotfile", type=str, nargs=1,
                    help="the plotfile you wish to plot")

args = parser.parse_args()

# read the data and convert to the primitive variables (and velocity
# magnitude)
sim = io.read(args.plotfile[0])
myd = sim.cc_data
myg = myd.grid

dens = myd.get_var("density")
xmom = myd.get_var("x-momentum")
ymom = myd.get_var("y-momentum")
ener = myd.get_var("energy")

rho = dens.v()

u = np.sqrt(xmom.v()**2 + ymom.v()**2)/rho

e = (ener.v() - 0.5*rho*u*u)/rho
gamma = myd.get_aux("gamma")
p = rho*e*(gamma - 1.0)

# get the exact solution
exact = np.loadtxt("cylindrical-sedov.out")

x_exact = exact[:, 1]
rho_exact = exact[:, 2]
u_exact = exact[:, 5]
p_exact = exact[:, 4]
# e_exact = exact[:, 4]

# radially bin

# see http://code.google.com/p/agpy/source/browse/trunk/agpy/radialprofile.py?r=317
# for inspiration

# first define the bins
rmin = 0
rmax = np.sqrt(myg.xmax**2 + myg.ymax**2)

nbins = np.int(np.sqrt(myg.nx**2 + myg.ny**2))

# bins holds the edges, so there is one more value than actual bin
# bin_centers holds the center value of the bin
bins = np.linspace(rmin, rmax, nbins+1)
bin_centers = 0.5*(bins[1:] + bins[:-1])

# radius of each zone
xcenter = 0.5*(myg.xmin + myg.xmax)
ycenter = 0.5*(myg.ymin + myg.ymax)

r = np.sqrt((myg.x2d[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1] - xcenter)**2 +
            (myg.y2d[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1] - ycenter)**2)

# bin the radii -- digitize returns an array with the same shape as
# the input array but with elements of the array specifying which bin
# that location belongs to.  The value of whichbin will be 1 if we are
# located in the bin defined by bins[0] to bins[1].  This means that
# there will be no 0s
whichbin = np.digitize(r.flat, bins)

# bincount counts the number of occurrences of each non-negative
# integer value in whichbin.  Each entry in ncount gives the number
# of occurrences of it in whichbin.  The length of ncount is
# set by the maximum value in whichbin
ncount = np.bincount(whichbin)

# now bin the associated data
rho_bin = np.zeros(len(ncount)-1, dtype=np.float64)
u_bin = np.zeros(len(ncount)-1, dtype=np.float64)
p_bin = np.zeros(len(ncount)-1, dtype=np.float64)

for n in range(len(ncount)):

    # remember that there are no whichbin == 0, since that corresponds
    # to the left edge.  So we want whichbin == 1 to correspond to the
    # first value of bin_centers (bin_centers[0])
    rho_bin[n-1] = np.sum(rho.flat[whichbin == n])/np.sum(ncount[n])
    u_bin[n-1] = np.sum(u.flat[whichbin == n])/np.sum(ncount[n])
    p_bin[n-1] = np.sum(p.flat[whichbin == n])/np.sum(ncount[n])

bin_centers = bin_centers[0:len(ncount)-1]

# plot
fig, axes = plt.subplots(nrows=3, ncols=1, num=1)

plt.rc("font", size=10)

ax = axes.flat[0]

ax.plot(x_exact, rho_exact, color="C0", zorder=-100, label="exact")
ax.scatter(bin_centers, rho_bin, marker="x", s=7, color="C1", label="simulation")

ax.set_ylabel(r"$\rho$")
ax.set_xlim(0, 0.6)
ax.legend(frameon=False, loc="best", fontsize="small")

ax = axes.flat[1]

ax.plot(x_exact, u_exact, color="C0", zorder=-100)
ax.scatter(bin_centers, u_bin, marker="x", s=7, color="C1")

ax.set_ylabel(r"$u$")
ax.set_xlim(0, 0.6)

ax = axes.flat[2]

ax.plot(x_exact, p_exact, color="C0", zorder=-100)
ax.scatter(bin_centers, p_bin, marker="x", s=7, color="C1")

ax.set_ylabel(r"$p$")
ax.set_xlim(0, 0.6)
ax.set_xlabel(r"r")

plt.subplots_adjust(hspace=0.25)

fig.set_size_inches(4.5, 8.0)

plt.savefig(args.o, bbox_inches="tight")
