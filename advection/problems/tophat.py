import sys
from util import runparams
import mesh.patch as patch
import numpy
from util import msg

def initData(myPatch):
    """ initialize the tophat advection problem """

    msg.bold("initializing the tophat advection problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(myPatch, patch.ccData2d):
        print "ERROR: patch invalid in tophat.py"
        print myPatch.__class__
        sys.exit()

    dens = myPatch.getVarPtr("density")

    xmin = runparams.getParam("mesh.xmin")
    xmax = runparams.getParam("mesh.xmax")

    ymin = runparams.getParam("mesh.ymin")
    ymax = runparams.getParam("mesh.ymax")

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)
    
    dens[:,:] = 0.0

    i = myPatch.grid.ilo
    while i <= myPatch.grid.ihi:

        j = myPatch.grid.jlo
        while j <= myPatch.grid.jhi:

            if (numpy.sqrt((myPatch.grid.x[i]-xctr)**2 + 
                           (myPatch.grid.y[j]-yctr)**2) < 0.1):
                dens[i,j] = 1.0
                    
            j += 1
        i += 1
    

    
                             
