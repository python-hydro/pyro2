import sys
from util import runparams
import mesh.patch as patch
import numpy
from util import msg

def initData(myPatch):
    """ initialize the sedov problem """

    msg.bold("initializing the sedov problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(myPatch, patch.ccData2d):
        print "ERROR: patch invalid in sedov.py"
        print myPatch.__class__
        sys.exit()

    # get the density, momenta, and energy as separate variables
    dens = myPatch.getVarPtr("density")
    xmom = myPatch.getVarPtr("x-momentum")
    ymom = myPatch.getVarPtr("y-momentum")
    ener = myPatch.getVarPtr("energy")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    dens[:,:] = 1.0
    xmom[:,:] = 0.0
    ymom[:,:] = 0.0

    E_sedov = 1.0

    r_init = runparams.getParam("sedov.r_init")

    gamma = runparams.getParam("eos.gamma")
    pi = 3.14159

    xmin = runparams.getParam("mesh.xmin")
    xmax = runparams.getParam("mesh.xmax")

    ymin = runparams.getParam("mesh.ymin")
    ymax = runparams.getParam("mesh.ymax")

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)


    # initialize the pressure by putting the explosion energy into a
    # volume of constant pressure.  Then compute the energy in a zone
    # from this.
    nsub = 4

    i = myPatch.grid.ilo
    while i <= myPatch.grid.ihi:

        j = myPatch.grid.jlo
        while j <= myPatch.grid.jhi:

            dist = numpy.sqrt((myPatch.grid.x[i] - xctr)**2 +
                              (myPatch.grid.y[j] - yctr)**2)

            if (dist < 2.0*r_init):
                pzone = 0.0

                ii = 0
                while ii < nsub:

                    jj = 0
                    while jj < nsub:

                        xsub = myPatch.grid.xl[i] + (myPatch.grid.dx/nsub)*(ii + 0.5)
                        ysub = myPatch.grid.yl[j] + (myPatch.grid.dy/nsub)*(jj + 0.5)

                        dist = numpy.sqrt((xsub - xctr)**2 + \
                                          (ysub - yctr)**2)

                        if dist <= r_init:
                            p = (gamma - 1.0)*E_sedov/(pi*r_init*r_init)
                        else:
                            p = 1.e-5

                        pzone += p

                        jj += 1
                    ii += 1

                p = pzone/(nsub*nsub)
            else:
                p = 1.e-5

            ener[i,j] = p/(gamma - 1.0)

            j += 1
        i += 1




