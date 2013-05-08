import sys
from util import runparams
import mesh.patch as patch
import numpy
import math
from util import msg

def initData(myPatch):
    """ initialize the rt problem """

    msg.bold("initializing the rt problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(myPatch, patch.ccData2d):
        print "ERROR: patch invalid in rt.py"
        print myPatch.__class__
        sys.exit()

    # get the density, momenta, and energy as separate variables
    dens = myPatch.getVarPtr("density")
    xmom = myPatch.getVarPtr("x-momentum")
    ymom = myPatch.getVarPtr("y-momentum")
    ener = myPatch.getVarPtr("energy")

    gamma = runparams.getParam("eos.gamma")

    grav = runparams.getParam("compressible.grav")


    dens1 = runparams.getParam("rt.dens1")
    dens2 = runparams.getParam("rt.dens2")
    p0 = runparams.getParam("rt.p0")
    amp = runparams.getParam("rt.amp")
    sigma = runparams.getParam("rt.sigma")


    # initialize the components, remember, that ener here is
    # rho*eint + 0.5*rho*v**2, where eint is the specific
    # internal energy (erg/g)
    xmom[:,:] = 0.0
    ymom[:,:] = 0.0
    dens[:,:] = 0.0

    # set the density to be stratified in the y-direction
    myg = myPatch.grid

    ycenter = 0.5*(myg.ymin + myg.ymax)

    p = myg.scratchArray()

    j = myg.jlo
    while j <= myg.jhi:
        if (myg.y[j] < ycenter):
            dens[:,j] = dens1
            p[:,j] = p0 + dens1*grav*myg.y[j]

        else:
            dens[:,j] = dens2
            p[:,j] = p0 + dens1*grav*ycenter + dens2*grav*(myg.y[j] - ycenter)


        j += 1


    ymom[:,:] = amp*numpy.sin(2.0*math.pi*myg.x2d/(myg.xmax-myg.xmin))*numpy.exp(-(myg.y2d-ycenter)**2/sigma**2)

    ymom *= dens*ymom

    # set the energy (P = cs2*dens)
    ener[:,:] = p[:,:]/(gamma - 1.0) + \
        0.5*(xmom[:,:]**2 + ymom[:,:]**2)/dens[:,:]


        
    

    
                             
