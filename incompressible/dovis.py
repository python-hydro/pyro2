import numpy
import pylab

from util import runparams

def dovis(myData, n):

    pylab.clf()

    pylab.rc("font", size=10)

    u = myData.getVarPtr("x-velocity")
    v = myData.getVarPtr("y-velocity")

    myg = myData.grid

    fig, axes = pylab.subplots(nrows=2, ncols=2, num=1)
    pylab.subplots_adjust(hspace=0.25)
    
    # x-velocity
    ax = axes.flat[0]
    
    img = ax.imshow(numpy.transpose(u[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
                    interpolation="nearest", origin="lower",
                    extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("x-velocity")

    pylab.colorbar(img, ax=ax)


    # y-velocity
    ax = axes.flat[1]
    
    img = ax.imshow(numpy.transpose(v[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
                    interpolation="nearest", origin="lower",
                    extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("y-velocity")

    pylab.colorbar(img, ax=ax)


    # vorticity
    ax = axes.flat[2]
    
    vort = myg.scratchArray()
    vort[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
        0.5*(v[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
             v[myg.ilo-1:myg.ihi,myg.jlo:myg.jhi+1])/myg.dx - \
         0.5*(u[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
              u[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi])/myg.dy


    img = ax.imshow(numpy.transpose(vort[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
                    interpolation="nearest", origin="lower",
                    extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"$\nabla \times U$")

    pylab.colorbar(img, ax=ax)


    # div U
    ax = axes.flat[3]
    
    divU = myg.scratchArray()

    divU[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
        0.5*(u[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] -
             u[myg.ilo-1:myg.ihi,myg.jlo:myg.jhi+1])/myg.dx + \
        0.5*(v[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -
             v[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi])/myg.dy


    img = ax.imshow(numpy.transpose(divU[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
                    interpolation="nearest", origin="lower",
                    extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"$\nabla \cdot U$")

    pylab.colorbar(img, ax=ax)




    pylab.figtext(0.05,0.0125, "t = %10.5f" % myData.t)

    pylab.draw()

