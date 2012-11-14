import pylab
import numpy
from matplotlib.font_manager import FontProperties
import eos

def dovis(myData, n):

    pylab.clf()

    pylab.rc("font", size=10)

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


    # figure out the geometry
    L_x = myData.grid.xmax - myData.grid.xmin
    L_y = myData.grid.ymax - myData.grid.ymin

    orientation = "vertical"
    shrink = 1.0

    sparseX = 0
    allYlabel = 1

    if (L_x > 2*L_y):

        # we want 4 rows:
        #  rho
        #  |U|
        #   p
        #   e
        fig, axes = pylab.subplots(nrows=4, ncols=1, num=1)
        orientation = "horizontal"
        if (L_x > 4*L_y):
            shrink = 0.75

    elif (L_y > 2*L_x):

        # we want 4 columns:
        # 
        #  rho  |U|  p  e
        fig, axes = pylab.subplots(nrows=1, ncols=4, num=1)        
        if (L_y >= 3*L_x):
            shrink = 0.5
            sparseX = 1
            allYlabel = 0

    else:
        # 2x2 grid of plots with 
        #
        #   rho   |u|
        #    p     e
        fig, axes = pylab.subplots(nrows=2, ncols=2, num=1)
        pylab.subplots_adjust(hspace=0.25)


    ax = axes.flat[0]

    img = ax.imshow(numpy.transpose(dens[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
                    interpolation="nearest", origin="lower",
                    extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"$\rho$")

    if sparseX:
        ax.xaxis.set_major_locator(pylab.MaxNLocator(3))

    pylab.colorbar(img, ax=ax, orientation=orientation, shrink=shrink)


    ax = axes.flat[1]


    img = ax.imshow(numpy.transpose(magvel[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
                    interpolation="nearest", origin="lower",
                    extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    ax.set_xlabel("x")
    if (allYlabel): ax.set_ylabel("y")
    ax.set_title("U")

    if sparseX:
        ax.xaxis.set_major_locator(pylab.MaxNLocator(3))

    pylab.colorbar(img, ax=ax, orientation=orientation, shrink=shrink)


    ax = axes.flat[2]

    img = ax.imshow(numpy.transpose(p[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
                    interpolation="nearest", origin="lower",
                    extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    ax.set_xlabel("x")
    if (allYlabel): ax.set_ylabel("y")
    ax.set_title("p")

    if sparseX:
        ax.xaxis.set_major_locator(pylab.MaxNLocator(3))

    pylab.colorbar(img, ax=ax, orientation=orientation, shrink=shrink)


    ax = axes.flat[3]

    img = ax.imshow(numpy.transpose(e[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
                    interpolation="nearest", origin="lower",
                    extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    ax.set_xlabel("x")
    if (allYlabel): ax.set_ylabel("y")
    ax.set_title("e")

    if sparseX:
        ax.xaxis.set_major_locator(pylab.MaxNLocator(3))

    pylab.colorbar(img, ax=ax, orientation=orientation, shrink=shrink)


    pylab.figtext(0.05,0.0125, "t = %10.5f" % myData.t)

    #fig.tight_layout()


    pylab.draw()
    #pylab.show()

