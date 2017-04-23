#!/usr/bin/env python3

from __future__ import print_function

import numpy as np
import sys

import mesh.patch
import mesh.array_indexer as ai

usage = """
      usage: ./compare.py file1 file2
"""

errors = {"gridbad": "grids don't agree",
          "namesbad": "variable lists don't agree",
          "varerr": "one or more variables don't agree"}


def compare(grid1, data1, grid2, data2):

    # compare the grids
    if not grid1 == grid2:
        return "gridbad"

    # compare the data
    if not data1.vars == data2.vars:
        return "namesbad"


    print(" ")
    print("variable comparisons:")

    result = 0

    for n in range(data1.nvar):

        # this is a hack for the changeover of the ordering of the data
        _d1 = data1.data
        if _d1.shape[2] < min(_d1.shape[0], _d1.shape[1]):
            d1 = ai.ArrayIndexer(_d1[:,:,n], grid=data1.grid)
        else:
            d1 = ai.ArrayIndexer(_d1[n,:,:], grid=data1.grid)

        _d2 = data2.data
        if _d2.shape[2] < min(_d2.shape[0], _d2.shape[1]):
            d2 = ai.ArrayIndexer(_d2[:,:,n], grid=data2.grid)
        else:
            d2 = ai.ArrayIndexer(_d2[n,:,:], grid=data2.grid)


        #d1 = data1.get_var(data1.vars[n])
        #d2 = data2.get_var(data2.vars[n])

        err = np.max(np.abs(d1.v() - d2.v()))

        print("%20s error = %20.10g" % (data1.vars[n], err))

        if not err == 0:
            result = "varerr"

    return result


if __name__ == "__main__":

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
