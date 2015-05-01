#!/usr/bin/env python

import numpy
import pylab
import sys
import mesh.patch as patch

# plot an output file using the solver's dovis script

def makeplot(myd, variable):

    pylab.figure(num=1, figsize=(1.28,1.28), dpi=100, facecolor='w')

    var = myd.getVarPtr(variable)
    myg = myd.grid

    img = pylab.imshow(numpy.transpose(var[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]),
                       interpolation="nearest", origin="lower",
                       extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    pylab.axis("off")

    pylab.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=1.0)

    pylab.savefig("plot.png")
    pylab.show()


if __name__== "__main__":

    print sys.argv

    file = sys.argv[1]
    variable = sys.argv[2]

    myg, myd = patch.read(file)

    makeplot(myd, variable)




