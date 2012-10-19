#!/usr/bin/env python

import numpy
import pylab
import sys
import mesh.patch as patch

# plot an output file using the solver's dovis script

def makeplot(myd, variable):

    pylab.figure(num=1, figsize=(6.5,5.25), dpi=100, facecolor='w')

    var = myd.getVarPtr(variable)
    myg = myd.grid

    img = pylab.imshow(numpy.transpose(var[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]),
                       interpolation="nearest", origin="lower",
                       extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    pylab.colorbar()

    #pylab.axis("off")

    pylab.xlabel("x")
    pylab.ylabel("y")

    pylab.savefig("plot.png", bbox_inches="tight")
    pylab.show()


if __name__== "__main__":

    print sys.argv

    file = sys.argv[1]
    variable = sys.argv[2]

    myg, myd = patch.read(file)

    makeplot(myd, variable)




