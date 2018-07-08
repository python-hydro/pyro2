#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
from util import io
import mesh.patch as patch

# plot a single variable from an output file
#
# Usage: ./plotvar.py filename variable


def makeplot(plotfile, variable, outfile,
             width=6.5, height=5.25,
             log=False, compact=False, quiet=False):

    sim = io.read(plotfile)

    if isinstance(sim, patch.CellCenterData2d):
        myd = sim
    else:
        myd = sim.cc_data
    myg = myd.grid

    plt.figure(num=1, figsize=(width, height), dpi=100, facecolor='w')

    var = myd.get_var(variable)

    if log:
        var = np.log10(var)

    plt.imshow(np.transpose(var.v()),
               interpolation="nearest", origin="lower",
               extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    if not compact:
        plt.colorbar()

        plt.xlabel("x")
        plt.ylabel("y")

    if compact:
        plt.axis("off")
        plt.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=1.0)
        plt.savefig(outfile)
    else:
        plt.savefig(outfile, bbox_inches="tight")

    if not quiet:
        plt.show()


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", type=str, default="plot.png",
                        metavar="plot.png", help="output file name")
    parser.add_argument("--log", action="store_true",
                        help="plot log of variable")
    parser.add_argument("--compact", action="store_true",
                        help="remove axes and border")
    parser.add_argument("--quiet", action="store_true",
                        help="don't show the figure")
    parser.add_argument("-W", type=float, default=6.5,
                        metavar="width", help="plot width (inches)")
    parser.add_argument("-H", type=float, default=5.25,
                        metavar="height", help="plot height (inches)")
    parser.add_argument("plotfile", type=str, nargs=1,
                        help="the plotfile you wish to plot")
    parser.add_argument("variable", type=str, nargs=1,
                        help="the name of the solver used to run the simulation")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()

    makeplot(args.plotfile[0], args.variable[0], args.o,
             width=args.W, height=args.H,
             log=args.log, compact=args.compact, quiet=args.quiet)
