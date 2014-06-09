import numpy
import pylab

def dovis(my_data, n):

    pylab.clf()

    phi = my_data.get_var("phi")

    myg = my_data.grid

    pylab.imshow(numpy.transpose(phi[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
                 interpolation="nearest", origin="lower",
                 extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    pylab.xlabel("x")
    pylab.ylabel("y")
    pylab.title("phi")

    pylab.colorbar()

    pylab.figtext(0.05,0.0125, "t = %10.5f" % my_data.t)

    pylab.draw()

