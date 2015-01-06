#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import getopt

import mesh.patch as patch

# plot an output file using the solver's dovis script

def makeplot(myd, var, outfile):

    a = myd.grid

    if var == "vort":
        vx = myd.get_var("x-velocity")
        vy = myd.get_var("y-velocity")

        v = myg.scratch_array()

        v[a.ilo:a.ihi+1,a.jlo:a.jhi+1] = \
             0.5*(vy[a.ilo+1:a.ihi+2,a.jlo:a.jhi+1] -
                  vy[a.ilo-1:a.ihi,a.jlo:a.jhi+1])/a.dx - \
             0.5*(vx[a.ilo:a.ihi+1,a.jlo+1:a.jhi+2] -
                  vx[a.ilo:a.ihi+1,a.jlo-1:a.jhi])/a.dy

    else:
        v = myd.get_var(var)



    plt.figure(num=1, figsize=(6.0,6.0), dpi=100, facecolor='w')

    plt.imshow(np.transpose(v[a.ilo:a.ihi+1,a.jlo:a.jhi+1]),
                 interpolation="nearest", origin="lower",
                 extent=[a.xmin, a.xmax, a.ymin, a.ymax])

    plt.axis("off")
    plt.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=1.0)
    plt.savefig(outfile)



def usage():
    usage="""
usage: plot.py [-h] [-o image.png] variable filename

positional arguments:
  variable      required inputs: variable to plot
  filename      required inputs: filename to read from

optional arguments:
  -h, --help    show this help message and exit
  -o image.png  output image name. The extension .png will generate a PNG
                file, .eps will generate an EPS file (default: plot.png).
"""
    print usage
    sys.exit()


if __name__== "__main__":

    try: opts, next = getopt.getopt(sys.argv[1:], "o:h:")
    except getopt.GetoptError:
        sys.exit("invalid calling sequence")

    outfile = "plot.png"

    for o, a in opts:
        if o == "-h": usage()
        if o == "-o": outfile = a

    try: var = next[0]
    except: usage()
        
    try: file = next[1]
    except: usage()

    myg, myd = patch.read(file)

    makeplot(myd, var, outfile)




