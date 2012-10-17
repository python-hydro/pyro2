#!/usr/bin/env python

import numpy
import mesh.patch as patch
import sys
import pylab

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

dens = myd.getVarPtr("density")
xmom = myd.getVarPtr("x-momentum")
ymom = myd.getVarPtr("y-momentum")
ener = myd.getVarPtr("energy")

rho = dens[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]

u = numpy.sqrt(xmom[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]**2 +
               ymom[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]**2)/rho

e = (ener[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] - 0.5*rho*u*u)/rho
gamma = myd.getAux("gamma")
p = rho*e*(gamma - 1.0)



# get the exact solution
exact = numpy.loadtxt("cylindrical-sedov.out")

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
rmax = numpy.sqrt(myg.xmax**2 + myg.ymax**2)

nbins = numpy.int(numpy.sqrt(myg.nx**2 + myg.ny**2))

# bins holds the edges, so there is one more value than actual bin
# bin_centers holds the center value of the bin
bins = numpy.linspace(rmin, rmax, nbins+1)
bin_centers = 0.5*(bins[1:] + bins[:-1])


# radius of each zone
xcenter = 0.5*(myg.xmin + myg.xmax)
ycenter = 0.5*(myg.ymin + myg.ymax)

r = numpy.sqrt( (myg.x2d[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] - xcenter)**2 +
                (myg.y2d[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] - ycenter)**2 )


# bin the radii -- digitize returns an array with the same shape as
# the input array but with elements of the array specifying which bin
# that location belongs to.  The value of whichbin will be 1 if we are
# located in the bin defined by bins[0] to bins[1].  This means that
# there will be no 0s
whichbin = numpy.digitize(r.flat, bins)

# bincount counts the number of occurrences of each non-negative
# integer value in whichbin.  Each entry in ncount gives the number
# of occurrences of it in whichbin.  The length of ncount is
# set by the maximum value in whichbin
ncount = numpy.bincount(whichbin)


# now bin the associated data
rho_bin = numpy.zeros(len(ncount)-1, dtype=numpy.float64)
u_bin   = numpy.zeros(len(ncount)-1, dtype=numpy.float64)
p_bin   = numpy.zeros(len(ncount)-1, dtype=numpy.float64)

n = 1
while (n < len(ncount)):

    # remember that there are no whichbin == 0, since that corresponds
    # to the left edge.  So we want whichbin == 1 to correspond to the
    # first value of bin_centers (bin_centers[0])
    rho_bin[n-1] = numpy.sum(rho.flat[whichbin==n])/numpy.sum(ncount[n])
    u_bin[n-1]   = numpy.sum(  u.flat[whichbin==n])/numpy.sum(ncount[n])
    p_bin[n-1]   = numpy.sum(  p.flat[whichbin==n])/numpy.sum(ncount[n])

    n += 1


bin_centers = bin_centers[0:len(ncount)-1]


# plot
fig, axes = pylab.subplots(nrows=3, ncols=1, num=1)        

pylab.rc("font", size=10)


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


pylab.subplots_adjust(hspace=0.25)


fig.set_size_inches(4.5,8.0)

pylab.savefig("sedov_compare.png", bbox_inches="tight")










