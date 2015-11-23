"""
compressible-specific boundary conditions.  Here, in particular, we
implement an HSE BC in the vertical direction.

Note: the pyro BC routines operate on a single variable at a time, so
some work will necessarily be repeated.

Also note: we may come in here with the aux_data (source terms), so
we'll do a special case for them

"""

import compressible.eos as eos
from util import msg

def user(bc_name, bc_edge, variable, ccdata):
    """
    A hydrostatic boundary.  This integrates the equation of HSE into
    the ghost cells to get the pressure and density under the assumption
    that the specific internal energy is constant.

    Upon exit, the ghost cells for the input variable will be set

    Parameters
    ----------
    bc_name : {'hse'}
        The descriptive name for the boundary condition -- this allows
        for pyro to have multiple types of user-supplied boundary
        conditions.  For this module, it needs to be 'hse'.
    bc_edge : {'ylb', 'yrb'}
        The boundary to update: ylb = lower y boundary; yrb = upper y
        boundary.
    variable : {'density', 'x-momentum', 'y-momentum', 'energy'}
        The variable whose ghost cells we are filling
    ccdata : CellCenterData2d object
        The data object

    """

    myg = ccdata.grid

    if bc_name == "hse":

        if bc_edge == "ylb":

            # lower y boundary

            # we will take the density to be constant, the velocity to
            # be outflow, and the pressure to be in HSE
            if variable in ["density", "x-momentum", "y-momentum", "ymom_src", "E_src"]:
                v = ccdata.get_var(variable)
                j = myg.jlo-1
                while j >= 0:
                    v.d[:,j] = v.d[:,myg.jlo]
                    j -= 1

            elif variable == "energy":
                dens = ccdata.get_var("density")
                xmom = ccdata.get_var("x-momentum")
                ymom = ccdata.get_var("y-momentum")
                ener = ccdata.get_var("energy")
                
                grav = ccdata.get_aux("grav")
                gamma = ccdata.get_aux("gamma")

                dens_base = dens.d[:,myg.jlo]
                ke_base = 0.5*(xmom.d[:,myg.jlo]**2 + ymom.d[:,myg.jlo]**2) / \
                    dens.d[:,myg.jlo]

                eint_base = (ener.d[:,myg.jlo] - ke_base)/dens.d[:,myg.jlo]
                pres_base = eos.pres(gamma, dens_base, eint_base)

                # we are assuming that the density is constant in this
                # formulation of HSE, so the pressure comes simply from
                # differencing the HSE equation
                j = myg.jlo-1
                while (j >= 0):
                    pres_below = pres_base - grav*dens_base*myg.dy
                    rhoe = eos.rhoe(gamma, pres_below)

                    ener.d[:,j] = rhoe + ke_base

                    pres_base = pres_below.copy()

                    j -= 1

            else:
                msg.fail("error: variable not defined")


        elif bc_edge == "yrb":

            # upper y boundary

            # we will take the density to be constant, the velocity to
            # be outflow, and the pressure to be in HSE
            if variable in ["density", "x-momentum", "y-momentum", "ymom_src", "E_src"]:
                v = ccdata.get_var(variable)
                for j in range(myg.jhi+1, myg.jhi+myg.ng+1):
                    v.d[:,j] = v.d[:,myg.jhi]

            elif variable == "energy":
                dens = ccdata.get_var("density")
                xmom = ccdata.get_var("x-momentum")
                ymom = ccdata.get_var("y-momentum")
                ener = ccdata.get_var("energy")
                
                grav = ccdata.get_aux("grav")
                gamma = ccdata.get_aux("gamma")

                dens_base = dens.d[:,myg.jhi]
                ke_base = 0.5*(xmom.d[:,myg.jhi]**2 + ymom.d[:,myg.jhi]**2) / \
                    dens.d[:,myg.jhi]

                eint_base = (ener.d[:,myg.jhi] - ke_base)/dens.d[:,myg.jhi]
                pres_base = eos.pres(gamma, dens_base, eint_base)

                # we are assuming that the density is constant in this
                # formulation of HSE, so the pressure comes simply from
                # differencing the HSE equation
                for j in range(myg.jhi+1, myg.jhi+myg.ng+1):
                    pres_above = pres_base + grav*dens_base*myg.dy
                    rhoe = eos.rhoe(gamma, pres_above)

                    ener.d[:,j] = rhoe + ke_base

                    pres_base = pres_above.copy()

            else:
                msg.fail("error: variable not defined")


        else:
            msg.fail("error: hse BC not supported for xlb or xrb")


    else:
        msg.fail("error: bc type %s not supported" % (bc_name) )
