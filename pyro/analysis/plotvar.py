#!/usr/bin/env python3
"""
plot a single variable from an output file

Usage: ./plotvar.py filename variable
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import pyro.util.io_pyro as io
from pyro.mesh import patch


def makeplot(plotfile, variable, outfile,
             width=6.5, height=5.25, dpi=100, notitle=False,
             log=False, compact=False, quiet=False):

    sim = io.read(plotfile)
    sim.cc_data.fill_BC_all()

    if isinstance(sim, patch.CellCenterData2d):
        myd = sim
    else:
        myd = sim.cc_data
    myg = myd.grid

    fig = plt.figure(num=1, figsize=(width, height), dpi=100, facecolor='w')
    ax = fig.add_subplot(111)

    if variable == "vort":
        vx = myd.get_var("x-velocity")
        vy = myd.get_var("y-velocity")

        var = myg.scratch_array()

        var[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1] = \
             0.5*(vy[myg.ilo+1:myg.ihi+2, myg.jlo:myg.jhi+1] -
                  vy[myg.ilo-1:myg.ihi, myg.jlo:myg.jhi+1])/myg.dx - \
             0.5*(vx[myg.ilo:myg.ihi+1, myg.jlo+1:myg.jhi+2] -
                  vx[myg.ilo:myg.ihi+1, myg.jlo-1:myg.jhi])/myg.dy

    else:
        var = myd.get_var(variable)

    if log:
        var = np.log10(var)

    img = ax.imshow(np.transpose(var.v()),
                    interpolation="nearest", origin="lower",
                    extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    if not compact:
        fig.colorbar(img)
        if not notitle:
            fig.suptitle(f"{variable}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    if compact:
        ax.axis("off")
        fig.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=1.0)
        fig.savefig(outfile, bbox_inches="tight")
    else:
        fig.tight_layout()
        fig.savefig(outfile, bbox_inches="tight", dpi=dpi)

    if not quiet:
        plt.show()


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", type=str, default="plot.png",
                        metavar="plot.png", help="output file name")
    parser.add_argument("--log", action="store_true",
                        help="plot log of variable")
    parser.add_argument("--notitle", action="store_true",
                        help="suppress the title at the top of the figure")
    parser.add_argument("--compact", action="store_true",
                        help="remove axes and border")
    parser.add_argument("--quiet", action="store_true",
                        help="don't show the figure")
    parser.add_argument("-W", type=float, default=6.5,
                        metavar="width", help="plot width (inches)")
    parser.add_argument("-H", type=float, default=5.25,
                        metavar="height", help="plot height (inches)")
    parser.add_argument("--dpi", type=int, default=100,
                        metavar="dpi", help="dots per inch")
    parser.add_argument("plotfile", type=str, nargs=1,
                        help="the plotfile you wish to plot")
    parser.add_argument("variable", type=str, nargs=1,
                        help="the name of the solver used to run the simulation")

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    makeplot(args.plotfile[0], args.variable[0], args.o,
             width=args.W, height=args.H, dpi=args.dpi, notitle=args.notitle,
             log=args.log, compact=args.compact, quiet=args.quiet)
