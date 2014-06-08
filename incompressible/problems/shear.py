"""
Initialize the doubly periodic shear layer (see, for example, Martin
and Colella, 2000, JCP, 163, 271).  This is run in a unit square
domain, with periodic boundary conditions on all sides.  Here, the
initial velocity is

              / tanh(rho_s (y-0.25))   if y <= 0.5
u(x,y,t=0) = <
              \ tanh(rho_s (0.75-y))   if y > 0.5


v(x,y,t=0) = delta_s sin(2 pi x)


"""

import sys
from util import runparams
import mesh.patch as patch
import numpy
from util import msg
import math

def initData(myPatch):
    """ initialize the incompressible shear problem """

    msg.bold("initializing the incompressible shear problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(myPatch, patch.CellCenterData2d):
        print myPatch.__class__
        msg.fail("ERROR: patch invalid in shear.py")


    # get the necessary runtime parameters
    rho_s = runparams.getParam("shear.rho_s")
    delta_s = runparams.getParam("shear.delta_s")

    
    # get the velocities
    u = myPatch.getVarPtr("x-velocity")
    v = myPatch.getVarPtr("y-velocity")

    myg = myPatch.grid

    if (myg.xmin != 0 or myg.xmax != 1 or
        myg.ymin != 0 or myg.ymax != 1):
        msg.fail("ERROR: domain should be a unit square")
        
    y_half = 0.5*(myg.ymin + myg.ymax)

    print 'y_half = ', y_half
    print 'delta_s = ', delta_s
    print 'rho_s = ', rho_s
    
    # there is probably an easier way to do this without loops, but
    # for now, we will just do an explicit loop.
    i = myg.ilo
    while i <= myg.ihi:

        j = myg.jlo
        while j <= myg.jhi:

            if (myg.y[j] <= y_half):
                u[i,j] = numpy.tanh(rho_s*(myg.y[j] - 0.25))
            else:
                u[i,j] = numpy.tanh(rho_s*(0.75 - myg.y[j]))
            
            v[i,j] = delta_s*numpy.sin(2.0*math.pi*myg.x[i])
            
            j += 1
        i += 1
        
    
    print "extrema: ", numpy.min(u.flat), numpy.max(u.flat)
    
    
                             
def finalize():
    """ print out any information to the user at the end of the run """
    pass
