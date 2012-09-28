import sys
from util import runparams
import mesh.patch as patch
import numpy
from util import msg
import math

def initData(myPatch):
    """ initialize the Kelvin-Helmholtz problem """

    msg.bold("initializing the sedov problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(myPatch, patch.ccData2d):
        print myPatch.__class__
        msg.fail("ERROR: patch invalid in sedov.py")


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

    rho_1 = runparams.getParam("kh.rho_1")
    v_1   = runparams.getParam("kh.v_1")
    rho_2 = runparams.getParam("kh.rho_2")
    v_2   = runparams.getParam("kh.v_2")

    gamma = runparams.getParam("eos.gamma")

    xmin = runparams.getParam("mesh.xmin")
    xmax = runparams.getParam("mesh.xmax")

    ymin = runparams.getParam("mesh.ymin")
    ymax = runparams.getParam("mesh.ymax")

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)

    L_x = xmax - xmin

    # initialize the pressure by putting the explosion energy into a
    # volume of constant pressure.  Then compute the energy in a zone
    # from this.
    nsub = 4

    i = myPatch.grid.ilo
    while i <= myPatch.grid.ihi:

        j = myPatch.grid.jlo
        while j <= myPatch.grid.jhi:

            if myPatch.grid.y[j] < yctr + 0.01*math.sin(10.0*math.pi*myPatch.grid.x[i]/L_x):

                # lower half
                dens[i,j] = rho_1
                xmom[i,j] = rho_1*v_1
                ymom[i,j] = 0.0
                
            else:

                # upper half
                dens[i,j] = rho_2
                xmom[i,j] = rho_2*v_2
                ymom[i,j] = 0.0

            p = 1.0
            ener[i,j] = p/(gamma - 1.0) + 0.5*(xmom[i,j]**2 + ymom[i,j]**2)/dens[i,j]

            j += 1
        i += 1




