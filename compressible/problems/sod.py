import sys
import mesh.patch as patch
import numpy
from util import msg

def initData(my_data):
    """ initialize the sod problem """

    msg.bold("initializing the sod problem...")

    rp = my_data.rp

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print "ERROR: patch invalid in sod.py"
        print my_data.__class__
        sys.exit()


    # get the sod parameters
    dens_left = rp.get_param("sod.dens_left")
    dens_right = rp.get_param("sod.dens_right")

    u_left = rp.get_param("sod.u_left")
    u_right = rp.get_param("sod.u_right")

    p_left = rp.get_param("sod.p_left")
    p_right = rp.get_param("sod.p_right")
    

    # get the density, momenta, and energy as separate variables
    dens = my_data.getVarPtr("density")
    xmom = my_data.getVarPtr("x-momentum")
    ymom = my_data.getVarPtr("y-momentum")
    ener = my_data.getVarPtr("energy")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    gamma = rp.get_param("eos.gamma")

    direction = rp.get_param("sod.direction")
    
    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)

    myg = my_data.grid
    
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
        
    

def finalize():
    """ print out any information to the user at the end of the run """

    msg = """
          The script analysis/sod_compare.py can be used to compare 
          this output to the exact solution.  Some sample exact solution
          data is present as analysis/sod-exact.out
          """

    print msg



                             
