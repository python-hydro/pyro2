#!/usr/bin/env python

import numpy
import mesh.patch as patch
import sys
import pylab
import diffusion.problems.gaussian as gaussian

usage = """
      Average the Gaussian diffusion profile and compare the radial profile 
      against the analytic solution.  Multiple files can be passed in.

      usage: ./gauss_diffusion_compare.py file1 file2 ...
"""

def abort(string):
    print string
    sys.exit(2)


def process(file):

    # read the data and convert to the primitive variables (and velocity
    # magnitude)
    myg, myd = patch.read(file)

    phi_t = myd.getVarPtr("phi")
    phi = phi_t[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]

    # get the problem parameters
    t_0     = myd.getAux("t_0")
    phi_0   = myd.getAux("phi_0")
    phi_max = myd.getAux("phi_max")
    k       = myd.getAux("k")

    t = myd.t

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


    # bin the radii -- digitize returns an array with the same shape
    # as the input array but with elements of the array specifying
    # which bin that location belongs to.  The value of whichbin will
    # be 1 if we are located in the bin defined by bins[0] to bins[1].
    # This means that there will be no 0s
    whichbin = numpy.digitize(r.flat, bins)

    # bincount counts the number of occurrences of each non-negative
    # integer value in whichbin.  Each entry in ncount gives the
    # number of occurrences of it in whichbin.  The length of ncount
    # is set by the maximum value in whichbin
    ncount = numpy.bincount(whichbin)


    # now bin the associated data
    phi_bin = numpy.zeros(len(ncount)-1, dtype=numpy.float64)

    n = 1
    while (n < len(ncount)):

        # remember that there are no whichbin == 0, since that
        # corresponds to the left edge.  So we want whichbin == 1 to
        # correspond to the first value of bin_centers
        # (bin_centers[0])
        phi_bin[n-1] = numpy.sum(phi.flat[whichbin==n])/numpy.sum(ncount[n])

        n += 1


    bin_centers = bin_centers[0:len(ncount)-1]


    # get the analytic solution
    phi_exact = gaussian.phi_analytic(bin_centers, t, t_0, k, phi_0, phi_max)

    
    return bin_centers, phi_exact, phi_bin
    


# main

if not len(sys.argv) >= 2:
    print usage
    sys.exit(2)


fig, ax = pylab.subplots(nrows=1, ncols=1, num=1)        
pylab.rc("font", size=10)


n = 1
while (n < len(sys.argv)):

    try: file = sys.argv[n]
    except:
        print usage
        sys.exit(2)

    bins, phi_exact, phi_bin = process(file)
        
    # plot
    ax.plot(bins, phi_exact, color="0.5")
    ax.scatter(bins, phi_bin, marker="x", s=7, color="r")

    n += 1

ax.set_xlim(0,0.4)
ax.set_xlabel(r"$r$")

ax.set_ylim(1.,1.05)
ax.set_ylabel(r"$\phi$")

fig.set_size_inches(5.0,5.0)

pylab.savefig("gauss_diffusion_compare.png", bbox_inches="tight")










