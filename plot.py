#!/usr/bin/env python

import numpy
import pylab
import sys
import getopt

import mesh.patch as patch

# plot an output file using the solver's dovis script

def makeplot(myd, solverName, outfile):

    exec 'import ' + solverName + ' as solver'

    pylab.figure(num=1, figsize=(8,4.5), dpi=100, facecolor='w')

    solver.dovis(myd, 0)
    pylab.savefig(outfile)
    pylab.show()


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
                                    
    try: solver = next[0]
    except: usage()
        
    try: file = next[1]
    except: usage()

    myg, myd = patch.read(file)

    makeplot(myd, solver, outfile)




