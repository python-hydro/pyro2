import sys
from util import runparams
import mesh.patch as patch
import numpy

def fillPatch(myPatch):
    """ initialize the smooth advection problem """

    print "initializing the smooth advection problem..."

    # make sure that we are passed a valid patch object
    if not isinstance(myPatch, patch.ccData2d):
        print "ERROR: patch invalid in smooth.py"
        print myPatch.__class__
        sys.exit()

    dens = myPatch.getVarPtr("density")

    xmin = runparams.getParam("mesh.xmin")
    xmax = runparams.getParam("mesh.xmax")

    ymin = runparams.getParam("mesh.ymin")
    ymax = runparams.getParam("mesh.ymax")

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)
    
    i = myPatch.grid.ilo
    while i <= myPatch.grid.ihi:

        j = myPatch.grid.jlo
        while j <= myPatch.grid.jhi:

            dens[i,j] = numpy.exp(-60.0*((myPatch.grid.x[i]-xctr)**2 + \
                                         (myPatch.grid.y[j]-yctr)**2))
                    
            j += 1
        i += 1
    

    
                             
