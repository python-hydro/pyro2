import sys
from util import runparams
import mesh.patch as patch
import numpy
from util import msg

def initData(myPatch):
    """ initialize the bubble problem """

    msg.bold("initializing the bubble problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(myPatch, patch.ccData2d):
        print "ERROR: patch invalid in bubble.py"
        print myPatch.__class__
        sys.exit()

    # get the density, momenta, and energy as separate variables
    dens = myPatch.getVarPtr("density")
    xmom = myPatch.getVarPtr("x-momentum")
    ymom = myPatch.getVarPtr("y-momentum")
    ener = myPatch.getVarPtr("energy")

    gamma = runparams.getParam("eos.gamma")

    grav = runparams.getParam("compressible.grav")

    scale_height = runparams.getParam("bubble.scale_height")
    dens_base = runparams.getParam("bubble.dens_base")
    dens_cutoff = runparams.getParam("bubble.dens_cutoff")

    x_pert = runparams.getParam("bubble.x_pert")
    y_pert = runparams.getParam("bubble.y_pert")
    r_pert = runparams.getParam("bubble.r_pert")
    pert_amplitude_factor = runparams.getParam("bubble.pert_amplitude_factor")

    # initialize the components, remember, that ener here is
    # rho*eint + 0.5*rho*v**2, where eint is the specific
    # internal energy (erg/g)
    xmom[:,:] = 0.0
    ymom[:,:] = 0.0
    dens[:,:] = dens_cutoff

    # set the density to be stratified in the y-direction
    myg = myPatch.grid

    j = myg.jlo
    while j <= myg.jhi:
        dens[:,j] = max(dens_base*numpy.exp(-myg.y[j]/scale_height),
                        dens_cutoff)
        j += 1

    cs2 = scale_height*abs(grav)

    # set the energy (P = cs2*dens)
    ener[:,:] = cs2*dens[:,:]/(gamma - 1.0) + \
                0.5*(xmom[:,:]**2 + ymom[:,:]**2)/dens[:,:]


    
    i = myg.ilo
    while i <= myg.ihi:

        j = myg.jlo
        while j <= myg.jhi:

            r = numpy.sqrt((myg.x[i] - x_pert)**2  + (myg.y[j] - y_pert)**2)

            if (r <= r_pert):
                # boost the specific internal energy, keeping the pressure
                # constant, by dropping the density
                eint = (ener[i,j] - 
                        0.5*(xmom[i,j]**2 - ymom[i,j]**2)/dens[i,j])/dens[i,j]

                pres = dens[i,j]*eint*(gamma - 1.0)

                eint = eint*pert_amplitude_factor
                dens[i,j] = pres/(eint*(gamma - 1.0))

                ener[i,j] = dens[i,j]*eint + \
                    0.5*(xmom[i,j]**2 + ymom[i,j]**2)/dens[i,j]

            j += 1
        i += 1
        
    

    
                             
