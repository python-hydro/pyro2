import sys
import mesh.patch as patch
import numpy
from util import msg
from util import runparams


def phi_analytic(dist, t, t_0, k, phi_1, phi_2):
    """ the analytic solution to the Gaussian diffusion problem """
    phi = (phi_2 - phi_1)*(t_0/(t + t_0))* \
        numpy.exp(-0.25*dist**2/(k*(t + t_0)) ) + phi_1
    return phi


def initData(myData):
    """ initialize the Gaussian diffusion problem """

    msg.bold("initializing the Gaussian diffusion problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(myData, patch.CellCenterData2d):
        print "ERROR: patch invalid in diffuse.py"
        print myData.__class__
        sys.exit()

    phi = myData.getVarPtr("phi")

    xmin = myData.grid.xmin
    xmax = myData.grid.xmax

    ymin = myData.grid.ymin
    ymax = myData.grid.ymax

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)
    
    k = runparams.getParam("diffusion.k")
    t_0 = runparams.getParam("gaussian.t_0")
    phi_max = runparams.getParam("gaussian.phi_max")
    phi_0   = runparams.getParam("gaussian.phi_0")

    dist = numpy.sqrt((myData.grid.x2d - xctr)**2 +
                      (myData.grid.y2d - yctr)**2)
    
    phi[:,:] = phi_analytic(dist, 0.0, t_0, k, phi_0, phi_max)

    # for later interpretation / analysis, store some auxillary data
    myData.setAux("k", k)
    myData.setAux("t_0", t_0)
    myData.setAux("phi_0", phi_0)
    myData.setAux("phi_max", phi_max)

    
                             
def finalize():
    """ print out any information to the user at the end of the run """

    msg = """
          The solution can be compared to the analytic solution with
          the script analysis/gauss_diffusion_compare.py
          """

    print msg

