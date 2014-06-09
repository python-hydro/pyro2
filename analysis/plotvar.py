#!/usr/bin/env python

import numpy
import pylab
import sys
import getopt

import mesh.patch as patch

# plot a single variable from an output file 
#
# Usage: ./plotvar.py filename variable

def makeplot(myd, variable, outfile):

    pylab.figure(num=1, figsize=(6.5,5.25), dpi=100, facecolor='w')

    var = myd.get_var(variable)
    myg = myd.grid

    img = pylab.imshow(numpy.transpose(var[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]),
                       interpolation="nearest", origin="lower",
                       extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    pylab.colorbar()

    pylab.xlabel("x")
    pylab.ylabel("y")

    pylab.savefig(outfile, bbox_inches="tight")
    pylab.show()


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




