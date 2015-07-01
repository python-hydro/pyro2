#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import getopt

import mesh.patch as patch

# plot a single variable from an output file
#
# Usage: ./plotvar.py filename variable

def makeplot(myd, variable, outfile):

    plt.figure(num=1, figsize=(6.5,5.25), dpi=100, facecolor='w')

    var = myd.get_var(variable)
    myg = myd.grid

    img = plt.imshow(np.transpose(var.v()),
                       interpolation="nearest", origin="lower",
                       extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    plt.colorbar()

    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig(outfile, bbox_inches="tight")
    plt.show()


def usage():
    usage="""
usage: plotvar.py [-h] [-o image.png] filename variable

positional arguments:
  filename      required inputs: filename to read from
  variable      required inputs: variable to plot from filename

optional arguments:
  -h, --help    show this help message and exit
  -o image.png  output image name. The extension .png will generate a PNG
                file, .eps will generate an EPS file (default: plot.png).
"""
    print usage
    sys.exit()


if __name__== "__main__":

    try: opts, next = getopt.getopt(sys.argv[1:], "o:h")
    except getopt.GetoptError:
        sys.exit("invalid calling sequence")

    outfile = "plot.png"

    for o, a in opts:
        if o == "-h": usage()
        if o == "-o": outfile = a

    try: file = next[0]
    except: usage()

    try: variable = next[1]
    except: usage()

    myg, myd = patch.read(file)

    makeplot(myd, variable, outfile)
