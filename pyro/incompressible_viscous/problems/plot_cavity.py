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


def makeplot(plotfile_name, outfile, Reynolds, streamline_density):
    """ plot the velocity magnitude and streamlines """

    sim = io.read(plotfile_name)
    myg = sim.cc_data.grid
    x = myg.x[myg.ilo:myg.ihi+1]
    y = myg.y[myg.jlo:myg.jhi+1]
    u = sim.cc_data.get_var("x-velocity")
    v = sim.cc_data.get_var("y-velocity")
    magvel = np.sqrt(u**2+v**2)

    _, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')

    img = ax.contourf(x, y, np.transpose(magvel.v()), 30, cmap="magma")
    ax.streamplot(x, y, np.transpose(u.v()), np.transpose(v.v()), broken_streamlines=False,
                  linewidth=0.5, arrowstyle="-", density=streamline_density, color='w')

    cbar = plt.colorbar(img, ax=ax)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("velocity magnitude", rotation=270)
    plt.xlabel("x")
    plt.ylabel("y")

    title = ""
    if Reynolds is not None:
        title += f"Re = {Reynolds}"

    # The characteristic timescale tau is the length of the box divided by the lid velocity.
    # For the default parameters, tau=1
    # title += rf", $t/\tau = $ {sim.cc_data.t:.1f}"
    plt.title(title)

    if outfile.endswith(".pdf"):
        plt.savefig(outfile, bbox_inches="tight")
    else:
        plt.savefig(outfile, bbox_inches="tight", dpi=500)
    plt.show()


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", type=str, default="plot.png",
                        metavar="plot.png", help="output file name")
    parser.add_argument("-Re", type=int,
                        metavar="Reynolds", help="Reynolds number (1/viscosity)")
    parser.add_argument("-d", type=float, default=0.25,
                        metavar="density", help="density of streamlines (0 to 1)")
    parser.add_argument("plotfile", type=str, nargs=1,
                        help="the plotfile you wish to plot")

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    makeplot(args.plotfile[0], args.o, args.Re, args.d)
