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
    magvel = u**2 + v**2   # temporarily |U|^2
    rhoe = (ener - 0.5*dens*magvel)

    magvel = numpy.sqrt(magvel)

    # access gamma from the object instead of using the EOS so we can
    # use dovis outside of a running simulation.
    gamma = myData.getAux("gamma")
    p = rhoe*(gamma - 1.0)

    e = rhoe/dens

    myg = myData.grid


    # 2x2 grid of plots with 
    #
    #   rho   |u|
    #    p     e

    fig, axes = pylab.subplots(nrows=2, ncols=2, num=1)

    ax = axes.flat[0]

    img = ax.imshow(numpy.transpose(dens[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
                    interpolation="nearest", origin="lower",
                    extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"$\rho$")

    pylab.colorbar(img, ax=ax)


    ax = axes.flat[1]


    img = ax.imshow(numpy.transpose(magvel[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
                    interpolation="nearest", origin="lower",
                    extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("U")

    pylab.colorbar(img, ax=ax)


    ax = axes.flat[2]

    img = ax.imshow(numpy.transpose(p[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
                    interpolation="nearest", origin="lower",
                    extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("p")

    pylab.colorbar(img, ax=ax)


    ax = axes.flat[3]

    img = ax.imshow(numpy.transpose(e[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
                    interpolation="nearest", origin="lower",
                    extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("e")

    pylab.colorbar(img, ax=ax)

    #fig.tight_layout()


    pylab.draw()
    #pylab.show()

    store = runparams.getParam("vis.store_images")

    if (store == 1):
        basename = runparams.getParam("io.basename")
        pylab.savefig(basename + "%4.4d" % (n) + ".png")

