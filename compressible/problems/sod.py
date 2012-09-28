import sys
from util import runparams
import mesh.patch as patch
import numpy
from util import msg

def initData(myPatch):
    """ initialize the sod problem """

    msg.bold("initializing the sod problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(myPatch, patch.ccData2d):
        print "ERROR: patch invalid in sod.py"
        print myPatch.__class__
        sys.exit()


    # get the sod parameters
    dens_left = runparams.getParam("sod.dens_left")
    dens_right = runparams.getParam("sod.dens_right")

    u_left = runparams.getParam("sod.u_left")
    u_right = runparams.getParam("sod.u_right")

    p_left = runparams.getParam("sod.p_left")
    p_right = runparams.getParam("sod.p_right")
    

    # get the density, momenta, and energy as separate variables
    dens = myPatch.getVarPtr("density")
    xmom = myPatch.getVarPtr("x-momentum")
    ymom = myPatch.getVarPtr("y-momentum")
    ener = myPatch.getVarPtr("energy")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    xmin = runparams.getParam("mesh.xmin")
    xmax = runparams.getParam("mesh.xmax")

    ymin = runparams.getParam("mesh.ymin")
    ymax = runparams.getParam("mesh.ymax")

    gamma = runparams.getParam("eos.gamma")

    direction = runparams.getParam("sod.direction")
    
    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)

    myg = myPatch.grid
    
    # there is probably an easier way to do this, but for now, we
    # will just do an explicit loop.  Also, we really want to set
    # the pressue and get the internal energy from that, and then
    # compute the total energy (which is what we store).  For now
    # we will just fake this.
    if direction == "x":

        i = myg.ilo
        while i <= myg.ihi:

            j = myg.jlo
            while j <= myg.jhi:

                if myg.x[i] <= xctr:
                    dens[i,j] = dens_left
                    xmom[i,j] = dens_left*u_left
                    ymom[i,j] = 0.0
                    ener[i,j] = p_left/(gamma - 1.0) + 0.5*xmom[i,j]*u_left
                
                else:
                    dens[i,j] = dens_right
                    xmom[i,j] = dens_right*u_right
                    ymom[i,j] = 0.0
                    ener[i,j] = p_right/(gamma - 1.0) + 0.5*xmom[i,j]*u_right
                    
                j += 1
            i += 1

    else:
        i = myg.ilo
        while i <= myg.ihi:

            j = myg.jlo
            while j <= myg.jhi:

                if myg.y[j] <= yctr:
                    dens[i,j] = dens_left
                    xmom[i,j] = 0.0
                    ymom[i,j] = dens_left*u_left
                    ener[i,j] = p_left/(gamma - 1.0) + 0.5*ymom[i,j]*u_left
                
                else:
                    dens[i,j] = dens_right
                    xmom[i,j] = 0.0
                    ymom[i,j] = dens_right*u_right
                    ener[i,j] = p_right/(gamma - 1.0) + 0.5*ymom[i,j]*u_right
                    
                j += 1
            i += 1
        
    

    
                             
