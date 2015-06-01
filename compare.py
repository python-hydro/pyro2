#!/usr/bin/env python

from __future__ import print_function

import numpy
import mesh.patch
import getopt
import sys

usage = """
      usage: ./compare.py file1 file2
"""

errors = {"gridbad": "grids don't agree",
          "namesbad": "variable lists don't agree",
          "varerr": "one or more variables don't agree"}


def compare(myg1, myd1, myg2, myd2):

    # compare the grids
    if not myg1 == myg2: 
        return "gridbad"

    # compare the data
    if not myd1.vars == myd2.vars:
        return "namesbad"


    print(" ")
    print("variable comparisons:")

    result = 0

    for n in range(myd1.nvar):

        d1 = myd1.get_var(myd1.vars[n])
        d2 = myd2.get_var(myd2.vars[n])

        err = numpy.max(numpy.abs(d1.v() - d2.v()))

        print("%20s error = %20.10g" % (myd1.vars[n], err))

        if not err == 0:
            result = "varerr"

    return result


if __name__== "__main__":

    if not len(sys.argv) == 3:
        print(usage)
        sys.exit(2)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    myg1, myd1 = mesh.patch.read(file1)
    myg2, myd2 = mesh.patch.read(file2)

    result = compare(myg1, myd1, myg2, myd2)

    if result == 0:
        print("SUCCESS: files agree")
    else:
        print("ERROR: ", errors[result])


    
