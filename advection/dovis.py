import numpy
import pylab

from util import runparams

def dovis(my_data, n):

    pylab.clf()

    dens = my_data.getVarPtr("density")

    myg = my_data.grid

    pylab.imshow(numpy.transpose(dens[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
                 interpolation="nearest", origin="lower",
                 extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    pylab.xlabel("x")
    pylab.ylabel("y")
    pylab.title("density")

    pylab.colorbar()

    pylab.figtext(0.05,0.0125, "t = %10.5f" % my_data.t)

    pylab.draw()

