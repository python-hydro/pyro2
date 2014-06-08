"""
Initialize a smooth incompressible convergence test.  Here, the
velocities are initialized as
                                                                               
u(x,y) = 1 - 2 cos(2 pi x) sin(2 pi y) 
v(x,y) = 1 + 2 sin(2 pi x) cos(2 pi y)

and the exact solution at some later time t is then 

u(x,y,t) = 1 - 2 cos(2 pi (x - t)) sin(2 pi (y - t))
v(x,y,t) = 1 + 2 sin(2 pi (x - t)) cos(2 pi (y - t))
p(x,y,t) = -cos(4 pi (x - t)) - cos(4 pi (y - t))

The numerical solution can be compared to the exact solution to
measure the convergence rate of the algorithm.

"""

import sys
from util import runparams
import mesh.patch as patch
import numpy
from util import msg
import math

def initData(myPatch):
    """ initialize the incompressible converge problem """

    msg.bold("initializing the incompressible converge problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(myPatch, patch.CellCenterData2d):
        print myPatch.__class__
        msg.fail("ERROR: patch invalid in converge.py")


    
    # get the velocities
    u = myPatch.getVarPtr("x-velocity")
    v = myPatch.getVarPtr("y-velocity")

    myg = myPatch.grid

    if (myg.xmin != 0 or myg.xmax != 1 or
        myg.ymin != 0 or myg.ymax != 1):
        msg.fail("ERROR: domain should be a unit square")
        

    u[:,:] = 1.0 - 2.0*numpy.cos(2.0*math.pi*myg.x2d)*numpy.sin(2.0*math.pi*myg.y2d)
    v[:,:] = 1.0 + 2.0*numpy.sin(2.0*math.pi*myg.x2d)*numpy.cos(2.0*math.pi*myg.y2d)

    
    
                             
def finalize():
    """ print out any information to the user at the end of the run """

    msg = """
          Comparisons to the analytic solution can be done using
          analysis/incomp_converge_error.py
          """

    print msg

