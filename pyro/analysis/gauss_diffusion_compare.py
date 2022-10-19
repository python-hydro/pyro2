#!/usr/bin/env python3


import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import pyro.diffusion.problems.gaussian as gaussian
import pyro.util.io_pyro as io

mpl.rcParams["text.usetex"] = True
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'

usage = """
      Average the Gaussian diffusion profile and compare the radial profile
      against the analytic solution.  Multiple files can be passed in.

      usage: ./gauss_diffusion_compare.py file1 file2 ...
"""


def abort(string):
    print(string)
    sys.exit(2)


def process(file):

    # read the data and convert to the primitive variables (and
    # velocity magnitude)
    sim = io.read(file)
    myd = sim.cc_data
    myg = myd.grid

    phi_t = myd.get_var("phi")
    phi = phi_t[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1]

    # get the problem parameters
    t_0 = myd.get_aux("t_0")
    phi_0 = myd.get_aux("phi_0")
    phi_max = myd.get_aux("phi_max")
    k = myd.get_aux("k")

    t = myd.t

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

    # bin the radii -- digitize returns an array with the same shape
    # as the input array but with elements of the array specifying
    # which bin that location belongs to.  The value of whichbin will
    # be 1 if we are located in the bin defined by bins[0] to bins[1].
    # This means that there will be no 0s
    whichbin = np.digitize(r.flat, bins)

    # bincount counts the number of occurrences of each non-negative
    # integer value in whichbin.  Each entry in ncount gives the
    # number of occurrences of it in whichbin.  The length of ncount
    # is set by the maximum value in whichbin
    ncount = np.bincount(whichbin)

    # now bin the associated data
    phi_bin = np.zeros(len(ncount)-1, dtype=np.float64)

    for n in range(len(ncount)):

        # remember that there are no whichbin == 0, since that
        # corresponds to the left edge.  So we want whichbin == 1 to
        # correspond to the first value of bin_centers
        # (bin_centers[0])
        phi_bin[n-1] = np.sum(phi.flat[whichbin == n])/np.sum(ncount[n])

    bin_centers = bin_centers[0:len(ncount)-1]

    # get the analytic solution
    phi_exact = gaussian.phi_analytic(bin_centers, t, t_0, k, phi_0, phi_max)

    return bin_centers, phi_exact, phi_bin


if __name__ == "__main__":

    if not len(sys.argv) >= 2:
        print(usage)
        sys.exit(2)

    fig, ax = plt.subplots(nrows=1, ncols=1, num=1)
    plt.rc("font", size=10)

    for n in range(1, len(sys.argv)):

        try:
            file = sys.argv[n]
        except IndexError:
            print(usage)
            sys.exit(2)

        bins, phi_exact, phi_bin = process(file)

        # plot
        ax.plot(bins, phi_exact, color="C0")
        ax.scatter(bins, phi_bin, marker="x", s=10, color="C1", zorder=100)

    ax.set_xlim(0, 0.4)
    ax.set_xlabel(r"$r$")

    ax.set_ylim(1., 1.05)
    ax.set_ylabel(r"$\phi$")

    fig.set_size_inches(6.0, 6.0)

    plt.savefig("gauss_diffusion_compare.png", bbox_inches="tight")
    plt.savefig("gauss_diffusion_compare.pdf", bbox_inches="tight", dpi=100)
