#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

import mesh.patch as patch

# plot an output file using the solver's dovis script

def makeplot(myd, solver_name, outfile, W, H):

    exec 'import ' + solver_name + ' as solver'

    sim = solver.Simulation(solver_name, None, None)
    sim.cc_data = myd

    plt.figure(num=1, figsize=(W,H), dpi=100, facecolor='w')

    sim.dovis()
    plt.savefig(outfile)
    plt.show()


def usage():
    usage="""
usage: plot.py [-h] [-o image.png] solver filename

positional arguments:
  solver        required inputs: solver name
  filename      required inputs: filename to read from

optional arguments:
  -h, --help    show this help message and exit
  -o image.png  output image name. The extension .png will generate a PNG
                file, .eps will generate an EPS file (default: plot.png).
  -W width      width in inches
  -H height     height in inches
"""
    print usage
    sys.exit()


if __name__== "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", type=str, default="plot.png", 
                        metavar="plot.png", help="output file name")
    parser.add_argument("-W", type=float, default=8.0,
                        metavar="width", help="width (in inches) of the plot (100 dpi)")
    parser.add_argument("-H", type=float, default=4.5,
                        metavar="height", help="height (in inches) of the plot (100 dpi)")
    parser.add_argument("solver", type=str, nargs=1,
                        help="the name of the solver used to run the simulation")
    parser.add_argument("plotfile", type=str, nargs=1,
                        help="the plotfile you wish to plot")

    args = parser.parse_args()

    myg, myd = patch.read(args.plotfile[0])

    makeplot(myd, args.solver[0], args.o, args.W, args.H)




