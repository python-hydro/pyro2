#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
import numpy as np

import pyro.util.io_pyro as io

# plot an output file using the solver's dovis script


def makeplot(myd, variable):

    plt.figure(num=1, figsize=(1.28, 1.28), dpi=100, facecolor='w')

    var = myd.get_var(variable)
    # u = myd.get_var("x-velocity")
    # v = myd.get_var("y-velocity")
    # var = np.sqrt(u*u + v*v)

    myg = myd.grid

    plt.imshow(np.transpose(var[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1]),
               interpolation="nearest", origin="lower",
               extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    plt.axis("off")

    plt.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=1.0)

    plt.savefig("plot.png")
    plt.show()


if __name__ == "__main__":

    print(sys.argv)

    file = sys.argv[1]
    variable = sys.argv[2]

    sim = io.read(file)

    makeplot(sim.cc_data, variable)
