#!/usr/bin/env python3

from __future__ import print_function

import numpy as np
import sys

import mesh.array_indexer as ai
from util import io

usage = """
      usage: ./compare.py file1 file2
"""

errors = {"gridbad": "grids don't agree",
          "namesbad": "variable lists don't agree",
          "varerr": "one or more variables don't agree"}


def compare(data1, data2):

    # compare the grids
    if not data1.grid == data2.grid:
        return "gridbad"

    # compare the data
    if not sorted(data1.names) == sorted(data2.names):
        return "namesbad"


    print(" ")
    print("variable comparisons:")

    result = 0

    for name in data1.names:

        d1 = data1.get_var(name)
        d2 = data2.get_var(name)

        err = np.max(np.abs(d1.v() - d2.v()))

        print("{:20s} error = {:20.10g}".format(name, err))

        if not err == 0:
            result = "varerr"

    return result


if __name__ == "__main__":

    if not len(sys.argv) == 3:
        print(usage)
        sys.exit(2)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    s1 = io.read(file1)
    s2 = io.read(file2)

    result = compare(s1.cc_data, s2.cc_data)

    if result == 0:
        print("SUCCESS: files agree")
    else:
        print("ERROR: ", errors[result])
