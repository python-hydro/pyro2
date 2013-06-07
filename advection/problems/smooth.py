import sys
import mesh.patch as patch
import numpy
from util import msg

def initData(myPatch):
    """ initialize the smooth advection problem """

    msg.bold("initializing the smooth advection problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(myPatch, patch.ccData2d):
        print "ERROR: patch invalid in smooth.py"
        print myPatch.__class__
        sys.exit()

    dens = myPatch.getVarPtr("density")

    xmin = myPatch.grid.xmin
    xmax = myPatch.grid.xmax

    ymin = myPatch.grid.ymin
    ymax = myPatch.grid.ymax

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)
    
    i = myPatch.grid.ilo
    while i <= myPatch.grid.ihi:

        j = myPatch.grid.jlo
        while j <= myPatch.grid.jhi:

            dens[i,j] = 1.0 + numpy.exp(-60.0*((myPatch.grid.x[i]-xctr)**2 + \
                                               (myPatch.grid.y[j]-yctr)**2))
                    
            j += 1
        i += 1
    

    
def finalize():
    """ print out any information to the user at the end of the run """
    pass
