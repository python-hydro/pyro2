import sys
from util import runparams
import mesh.patch as patch
import numpy
from util import msg

def initData(myPatch):
    """ initialize the quadrant problem """

    msg.bold("initializing the quadrant problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(myPatch, patch.ccData2d):
        print "ERROR: patch invalid in quad.py"
        print myPatch.__class__
        sys.exit()

    # get the density, momenta, and energy as separate variables
    dens = myPatch.getVarPtr("density")
    xmom = myPatch.getVarPtr("x-momentum")
    ymom = myPatch.getVarPtr("y-momentum")
    ener = myPatch.getVarPtr("energy")

    # initialize the components, remember, that ener here is
    # rho*eint + 0.5*rho*v**2, where eint is the specific
    # internal energy (erg/g)
    r1 = runparams.getParam("quadrant.rho1")
    u1 = runparams.getParam("quadrant.u1")
    v1 = runparams.getParam("quadrant.v1")
    p1 = runparams.getParam("quadrant.p1")

    r2 = runparams.getParam("quadrant.rho2")
    u2 = runparams.getParam("quadrant.u2")
    v2 = runparams.getParam("quadrant.v2")
    p2 = runparams.getParam("quadrant.p2")

    r3 = runparams.getParam("quadrant.rho3")
    u3 = runparams.getParam("quadrant.u3")
    v3 = runparams.getParam("quadrant.v3")
    p3 = runparams.getParam("quadrant.p3")

    r4 = runparams.getParam("quadrant.rho4")
    u4 = runparams.getParam("quadrant.u4")
    v4 = runparams.getParam("quadrant.v4")
    p4 = runparams.getParam("quadrant.p4")

    cx = runparams.getParam("quadrant.cx")
    cy = runparams.getParam("quadrant.cy")
    
    gamma = runparams.getParam("eos.gamma")
    
    xmin = runparams.getParam("mesh.xmin")
    xmax = runparams.getParam("mesh.xmax")

    ymin = runparams.getParam("mesh.ymin")
    ymax = runparams.getParam("mesh.ymax")


    # there is probably an easier way to do this, but for now, we
    # will just do an explicit loop.  Also, we really want to set
    # the pressue and get the internal energy from that, and then
    # compute the total energy (which is what we store).  For now
    # we will just fake this
    
    myg = myPatch.grid

    i = myg.ilo
    while i <= myg.ihi:

        j = myg.jlo
        while j <= myg.jhi:

            if (myg.x[i] >= cx and myg.y[j] >= cy):

                # quadrant 1
                dens[i,j] = r1
                xmom[i,j] = r1*u1
                ymom[i,j] = r1*v1
                ener[i,j] = p1/(gamma - 1.0) + 0.5*r1*(u1*u1 + v1*v1)
                
            elif (myg.x[i] < cx and myg.y[j] >= cy):

                # quadrant 2
                dens[i,j] = r2
                xmom[i,j] = r2*u2
                ymom[i,j] = r2*v2
                ener[i,j] = p2/(gamma - 1.0) + 0.5*r2*(u2*u2 + v2*v2)

            elif (myg.x[i] < cx and myg.y[j] < cy):

                # quadrant 3
                dens[i,j] = r3
                xmom[i,j] = r3*u3
                ymom[i,j] = r3*v3
                ener[i,j] = p3/(gamma - 1.0) + 0.5*r3*(u3*u3 + v3*v3)

            elif (myg.x[i] >= cx and myg.y[j] < cy):

                # quadrant 4
                dens[i,j] = r4
                xmom[i,j] = r4*u4
                ymom[i,j] = r4*v4
                ener[i,j] = p4/(gamma - 1.0) + 0.5*r4*(u4*u4 + v4*v4)

            j += 1
        i += 1
        
    

    
                             
