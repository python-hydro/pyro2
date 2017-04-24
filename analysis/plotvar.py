#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
from util import io

# plot a single variable from an output file
#
# Usage: ./plotvar.py filename variable

def makeplot(plotfile, variable, outfile):

    sim = io.read(plotfile)
    myd = sim.cc_data
    myg = myd.grid

    plt.figure(num=1, figsize=(6.5,5.25), dpi=100, facecolor='w')

    var = myd.get_var(variable)

    img = plt.imshow(np.transpose(var.v()),
                     interpolation="nearest", origin="lower",
                     extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    plt.colorbar()

    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig(outfile, bbox_inches="tight")
    plt.show()


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", type=str, default="plot.png",
                        metavar="plot.png", help="output file name")
    parser.add_argument("plotfile", type=str, nargs=1,
                        help="the plotfile you wish to plot")
    parser.add_argument("variable", type=str, nargs=1,
                        help="the name of the solver used to run the simulation")

    args = parser.parse_args()

    return args


if __name__== "__main__":

    args = get_args()

    makeplot(args.plotfile[0], args.variable[0], args.o)
