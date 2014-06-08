"""
compressible-specific boundary conditions.  Here, in particular, we
implement an HSE BC in the vertical direction.

Note: the pyro BC routines operate on a single variable at a time, so
some work will necessarily be repeated.
"""

import eos
from util import msg, runparams

def user(bcName, bcEdge, variable, my_data):

    dens = my_data.getVarPtr("density")
    xmom = my_data.getVarPtr("x-momentum")
    ymom = my_data.getVarPtr("y-momentum")
    ener = my_data.getVarPtr("energy")

    grav = runparams.getParam("compressible.grav")

    myg = my_data.grid

    if (bcName == "hse"):
        
        if (bcEdge == "ylb"):

            # lower y boundary
            
            # we will take the density to be constant, the velocity to
            # be outflow, and the pressure to be in HSE
            if variable == "density":
                j = myg.jlo-1
                while (j >= 0):
                    dens[:,j] = dens[:,myg.jlo]
                    j -= 1

            elif variable == "x-momentum":
                j = myg.jlo-1
                while (j >= 0):
                    xmom[:,j] = xmom[:,myg.jlo]                
                    j -= 1

            elif variable == "y-momentum":
                j = myg.jlo-1
                while (j >= 0):
                    ymom[:,j] = ymom[:,myg.jlo]                
                    j -= 1

            elif variable == "energy":
                dens_base = dens[:,myg.jlo]
                ke_base = 0.5*(xmom[:,myg.jlo]**2 + ymom[:,myg.jlo]**2) / \
                    dens[:,myg.jlo]

                eint_base = (ener[:,myg.jlo] - ke_base)/dens[:,myg.jlo]
                pres_base = eos.pres(dens_base, eint_base)
                
                # we are assuming that the density is constant in this
                # formulation of HSE, so the pressure comes simply from
                # differencing the HSE equation
                j = myg.jlo-1
                while (j >= 0):
                    pres_below = pres_base - grav*dens_base*myg.dy
                    rhoe = eos.rhoe(pres_below)

                    ener[:,j] = rhoe + ke_base

                    pres_base = pres_below.copy()

                    j -= 1

            else:
                msg.fail("error: variable not defined")


        elif (bcEdge == "yrb"):

            # upper y boundary
            
            # we will take the density to be constant, the velocity to
            # be outflow, and the pressure to be in HSE
            if variable == "density":
                j = myg.jhi+1
                while (j <= myg.jhi+myg.ng):
                    dens[:,j] = dens[:,myg.jhi]
                    j += 1

            elif variable == "x-momentum":
                j = myg.jhi+1
                while (j <= myg.jhi+myg.ng):
                    xmom[:,j] = xmom[:,myg.jhi]                
                    j += 1

            elif variable == "y-momentum":
                j = myg.jhi+1
                while (j <= myg.jhi+myg.ng):
                    ymom[:,j] = ymom[:,myg.jhi]                
                    j += 1

            elif variable == "energy":
                dens_base = dens[:,myg.jhi]
                ke_base = 0.5*(xmom[:,myg.jhi]**2 + ymom[:,myg.jhi]**2) / \
                    dens[:,myg.jhi]

                eint_base = (ener[:,myg.jhi] - ke_base)/dens[:,myg.jhi]
                pres_base = eos.pres(dens_base, eint_base)
                
                # we are assuming that the density is constant in this
                # formulation of HSE, so the pressure comes simply from
                # differencing the HSE equation
                j = myg.jhi+1
                while (j <= myg.jhi+myg.ng):
                    pres_above = pres_base + grav*dens_base*myg.dy
                    rhoe = eos.rhoe(pres_above)

                    ener[:,j] = rhoe + ke_base

                    pres_base = pres_above.copy()

                    j += 1

            else:
                msg.fail("error: variable not defined")


        else:
            msg.fail("error: hse BC not supported for xlb or xrb")


    else:
        msg.fail("error: bc type %s not supported" % (bcName) )
