#!/usr/bin/env python3

import matplotlib.pyplot as plt
import argparse
import importlib
import mesh.patch as patch
import util.io as io

# plot an output file using the solver's dovis script

def makeplot(plotfile_name, outfile, width, height):
    """ plot the data in a plotfile using the solver's vis() method """

    solver_name, problem_name, myd = io.read(plotfile_name)

    solver = importlib.import_module(solver_name)

    sim = solver.Simulation(solver_name, None, None)
    sim.cc_data = myd

    plt.figure(num=1, figsize=(width, height), dpi=100, facecolor='w')

    sim.dovis()
    if outfile.endswith(".pdf"):
        plt.savefig(outfile, bbox_inches="tight")
    else:
        plt.savefig(outfile)
    plt.show()


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", type=str, default="plot.png",
                        metavar="plot.png", help="output file name")
    parser.add_argument("-W", type=float, default=8.0,
                        metavar="width", help="width (in inches) of the plot (100 dpi)")
    parser.add_argument("-H", type=float, default=4.5,
                        metavar="height", help="height (in inches) of the plot (100 dpi)")
    parser.add_argument("plotfile", type=str, nargs=1,
                        help="the plotfile you wish to plot")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()

    makeplot(args.plotfile[0], args.o, args.W, args.H)
