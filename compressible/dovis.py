import pylab
import numpy
from util import runparams
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

    if (L_x > 2*L_y):

        # we want 4 rows:
        #  rho
        #  |U|
        #   p
        #   e
        fig, axes = pylab.subplots(nrows=4, ncols=1, num=1)
        orientation = "horizontal"

    elif (L_y > 2*L_x):

        # we want 4 columns:
        # 
        #  rho  |U|  p  e
        fig, axes = pylab.subplots(nrows=1, ncols=4, num=1)        
        if (L_y > 4*L_x):
            shrink = 0.5

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

    pylab.colorbar(img, ax=ax, orientation=orientation, shrink=shrink)


    ax = axes.flat[1]


    img = ax.imshow(numpy.transpose(magvel[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
                    interpolation="nearest", origin="lower",
                    extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("U")

    pylab.colorbar(img, ax=ax, orientation=orientation, shrink=shrink)


    ax = axes.flat[2]

    img = ax.imshow(numpy.transpose(p[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
                    interpolation="nearest", origin="lower",
                    extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("p")

    pylab.colorbar(img, ax=ax, orientation=orientation, shrink=shrink)


    ax = axes.flat[3]

    img = ax.imshow(numpy.transpose(e[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1]), 
                    interpolation="nearest", origin="lower",
                    extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("e")

    pylab.colorbar(img, ax=ax, orientation=orientation, shrink=shrink)


    pylab.figtext(0.05,0.0125, "t = %10.5f" % myData.t)

    #fig.tight_layout()


    pylab.draw()
    #pylab.show()

    store = runparams.getParam("vis.store_images")

    if (store == 1):
        basename = runparams.getParam("io.basename")
        pylab.savefig(basename + "%4.4d" % (n) + ".png")

