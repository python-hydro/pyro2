"""
compressible-specific boundary conditions.  Here, in particular, we
implement an HSE BC in the vertical direction.

Note: the pyro BC routines operate on a single variable at a time, so
some work will necessarily be repeated.
"""

import compressible.eos as eos
from util import msg

def user(bc_name, bc_edge, variable, my_data):
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
    my_data : CellCenterData2d object
        The data object

    """
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    grav = my_data.get_aux("grav")
    gamma = my_data.get_aux("gamma")

    myg = my_data.grid

    if bc_name == "hse":

        if bc_edge == "ylb":

            # lower y boundary

            # we will take the density to be constant, the velocity to
            # be outflow, and the pressure to be in HSE
            if variable == "density":
                j = myg.jlo-1
                while j >= 0:
                    dens.d[:,j] = dens.d[:,myg.jlo]
                    j -= 1

            elif variable == "x-momentum":
                j = myg.jlo-1
                while j >= 0:
                    xmom.d[:,j] = xmom.d[:,myg.jlo]
                    j -= 1

            elif variable == "y-momentum":
                j = myg.jlo-1
                while j >= 0:
                    ymom.d[:,j] = ymom.d[:,myg.jlo]
                    j -= 1

            elif variable == "energy":
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
            if variable == "density":
                for j in range(myg.jhi+1, myg.jhi+myg.ng+1):
                    dens.d[:,j] = dens.d[:,myg.jhi]

            elif variable == "x-momentum":
                for j in range(myg.jhi+1, myg.jhi+myg.ng+1):
                    xmom.d[:,j] = xmom.d[:,myg.jhi]

            elif variable == "y-momentum":
                for j in range(myg.jhi+1, myg.jhi+myg.ng+1):
                    ymom.d[:,j] = ymom.d[:,myg.jhi]

            elif variable == "energy":
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
