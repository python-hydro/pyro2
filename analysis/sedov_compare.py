#!/usr/bin/env python

import numpy as np
import mesh.patch as patch
import sys
import matplotlib.pyplot as plt

usage = """
      compare the output for a Sedov problem with the exact solution contained
      in cylindrical-sedov.out.  To do this, we need to bin the Sedov data
      into radial bins.

      usage: ./sedov_compare.py file
"""

def abort(string):
    print string
    sys.exit(2)


if not len(sys.argv) == 2:
    print usage
    sys.exit(2)


try: file1 = sys.argv[1]
except:
    print usage
    sys.exit(2)


# read the data and convert to the primitive variables (and velocity
# magnitude)
myg, myd = patch.read(file1)

dens = myd.get_var("density")
xmom = myd.get_var("x-momentum")
ymom = myd.get_var("y-momentum")
ener = myd.get_var("energy")

rho = dens[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]

u = np.sqrt(xmom[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]**2 +
               ymom[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]**2)/rho

e = (ener[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] - 0.5*rho*u*u)/rho
gamma = myd.get_aux("gamma")
p = rho*e*(gamma - 1.0)



# get the exact solution
exact = np.loadtxt("cylindrical-sedov.out")

x_exact   = exact[:,1]
rho_exact = exact[:,2]
u_exact   = exact[:,5]
p_exact   = exact[:,4]
#e_exact   = exact[:,4]


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

r = np.sqrt( (myg.x2d[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] - xcenter)**2 +
                (myg.y2d[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] - ycenter)**2 )


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
u_bin   = np.zeros(len(ncount)-1, dtype=np.float64)
p_bin   = np.zeros(len(ncount)-1, dtype=np.float64)

for n in range(len(ncount)):

    # remember that there are no whichbin == 0, since that corresponds
    # to the left edge.  So we want whichbin == 1 to correspond to the
    # first value of bin_centers (bin_centers[0])
    rho_bin[n-1] = np.sum(rho.flat[whichbin==n])/np.sum(ncount[n])
    u_bin[n-1]   = np.sum(  u.flat[whichbin==n])/np.sum(ncount[n])
    p_bin[n-1]   = np.sum(  p.flat[whichbin==n])/np.sum(ncount[n])


bin_centers = bin_centers[0:len(ncount)-1]


# plot
fig, axes = plt.subplots(nrows=3, ncols=1, num=1)

plt.rc("font", size=10)


ax = axes.flat[0]

ax.plot(x_exact, rho_exact)
ax.scatter(bin_centers, rho_bin, marker="x", s=7, color="r")

ax.set_ylabel(r"$\rho$")
ax.set_xlim(0,0.6)

ax = axes.flat[1]

ax.plot(x_exact, u_exact)
ax.scatter(bin_centers, u_bin, marker="x", s=7, color="r")

ax.set_ylabel(r"$u$")
ax.set_xlim(0,0.6)

ax = axes.flat[2]

ax.plot(x_exact, p_exact)
ax.scatter(bin_centers, p_bin, marker="x", s=7, color="r")

ax.set_ylabel(r"$p$")
ax.set_xlim(0,0.6)
ax.set_xlabel(r"r")


plt.subplots_adjust(hspace=0.25)


fig.set_size_inches(4.5,8.0)

plt.savefig("sedov_compare.png", bbox_inches="tight")
