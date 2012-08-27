import pylab
import numpy

def dovis(myData):

    pylab.clf()

    dens = myData.getVarPtr("density")

    myg = myData.grid

    pylab.imshow(numpy.transpose(dens[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
                 interpolation="nearest", origin="lower",
                 extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    pylab.draw()

