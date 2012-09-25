import pylab
import numpy
from util import runparams
import eos

def dovis(myData, n):

    pylab.clf()

    dens = myData.getVarPtr("density")
    xmom = myData.getVarPtr("x-momentum")
    ymom = myData.getVarPtr("y-momentum")
    ener = myData.getVarPtr("energy")

    # get the velocities
    u = xmom/dens
    v = ymom/dens

    # get the pressure
    e = (ener - 0.5*(xmom**2 + ymom**2)/dens)/dens

    p = eos.pres(dens, e)

    myg = myData.grid


    # 2x2 grid of plots with 
    #
    #   rho   |u|
    #    p     e

    pylab.subplot(221)

    # pylab.imshow(numpy.transpose(dens[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
    #              interpolation="nearest", origin="lower",
    #              extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    pylab.imshow(dens[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1], 
                 interpolation="nearest", origin="lower",
                 extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    pylab.xlabel("x")
    pylab.ylabel("y")
    pylab.title("rho")

    pylab.colorbar()


    pylab.subplot(222)

    magvel = numpy.sqrt(u**2 + v**2)

    # pylab.imshow(numpy.transpose(magvel[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
    #              interpolation="nearest", origin="lower",
    #              extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    pylab.imshow(magvel[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1], 
                 interpolation="nearest", origin="lower",
                 extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    pylab.xlabel("x")
    pylab.ylabel("y")
    pylab.title("U")

    pylab.colorbar()


    pylab.subplot(223)

    # pylab.imshow(numpy.transpose(p[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
    #              interpolation="nearest", origin="lower",
    #              extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    pylab.imshow(p[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1], 
                 interpolation="nearest", origin="lower",
                 extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    pylab.xlabel("x")
    pylab.ylabel("y")
    pylab.title("p")

    pylab.colorbar()


    pylab.subplot(224)

    # pylab.imshow(numpy.transpose(e[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
    #              interpolation="nearest", origin="lower",
    #              extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    pylab.imshow(e[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1], 
                 interpolation="nearest", origin="lower",
                 extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    pylab.xlabel("x")
    pylab.ylabel("y")
    pylab.title("e")

    pylab.colorbar()

    #    pylab.tight_layout()


    pylab.draw()
    pylab.show()

    store = runparams.getParam("vis.store_images")

    if (store == 1):
        basename = runparams.getParam("io.basename")
        pylab.savefig(basename + "%4.4d" % (n) + ".png")

