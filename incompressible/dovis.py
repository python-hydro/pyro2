import pylab
import numpy
from util import runparams

def dovis(myData, n):

    pylab.clf()

    u = myData.getVarPtr("x-velocity")
    v = myData.getVarPtr("y-velocity")

    myg = myData.grid

    fig, axes = pylab.subplots(nrows=1, ncols=2, num=1)
    
    # x-velocity
    ax = axes.flat[0]
    
    img = ax.imshow(numpy.transpose(u[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
                    interpolation="nearest", origin="lower",
                    extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("x-velocity")

    pylab.colorbar(img, ax=ax, orientation="horizontal")


    # y-velocity
    ax = axes.flat[1]
    
    img = ax.imshow(numpy.transpose(v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
                    interpolation="nearest", origin="lower",
                    extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("y-velocity")

    pylab.colorbar(img, ax=ax, orientation="horizontal")

    pylab.figtext(0.05,0.0125, "t = %10.5f" % myData.t)

    pylab.draw()

