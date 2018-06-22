"""
compressible-specific boundary conditions.  Here, in particular, we
implement an HSE BC in the vertical direction.

Note: the pyro BC routines operate on a single variable at a time, so
some work will necessarily be repeated.

Also note: we may come in here with the aux_data (source terms), so
we'll do a special case for them

"""

import compressible_sr.eos as eos
import compressible_sr.unsplit_fluxes as flx
from util import msg
import numpy as np


def user(bc_name, bc_edge, variable, ccdata, ivars):
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
            if variable in ["density", "x-momentum", "y-momentum", "ymom_src", "E_src", "fuel", "ash"]:
                v = ccdata.get_var(variable)
                j = myg.jlo-1
                while j >= 0:
                    v[:, j] = v[:, myg.jlo]
                    j -= 1

            elif variable == "energy":
                # dens = ccdata.get_var("density")
                # xmom = ccdata.get_var("x-momentum")
                # ymom = ccdata.get_var("y-momentum")
                ener = ccdata.get_var("energy")

                grav = ccdata.get_aux("grav")
                gamma = ccdata.get_aux("gamma")

                q = flx.cons_to_prim_wrapper(ccdata.data, gamma, ivars, myg)

                rho = q[:,:,ivars.irho]
                u = q[:,myg.jlo,ivars.iu]
                v = q[:,myg.jlo,ivars.iv]
                p = q[:,:,ivars.ip]

                dens_base = rho[:, myg.jlo]
                # ke_base = 0.5*(u[:, myg.jlo]**2 + v[:, myg.jlo]**2) * rho[:, myg.jlo]

                # eint_base = (ener[:, myg.jlo] - ke_base)/dens[:, myg.jlo]
                # pres_base = eos.pres(gamma, dens_base, eint_base)
                pres_base = p[:, myg.jlo]

                # we are assuming that the density is constant in this
                # formulation of HSE, so the pressure comes simply from
                # differencing the HSE equation
                W = np.sqrt(1-u**2-v**2)
                j = myg.jlo-1
                while j >= 0:
                    pres_below = pres_base - grav*dens_base*myg.dy
                    rhoh = eos.rhoh_from_rho_p(gamma, dens_base, pres_below)

                    ener[:, j] = rhoh*W**2 - pres_below - dens_base

                    pres_base = pres_below.copy()

                    j -= 1

            else:
                raise NotImplementedError("variable not defined")

        elif bc_edge == "yrb":

            # upper y boundary

            # we will take the density to be constant, the velocity to
            # be outflow, and the pressure to be in HSE
            if variable in ["density", "x-momentum", "y-momentum", "ymom_src", "E_src", "fuel", "ash"]:
                v = ccdata.get_var(variable)
                for j in range(myg.jhi+1, myg.jhi+myg.ng+1):
                    v[:, j] = v[:, myg.jhi]

            elif variable == "energy":
                # dens = ccdata.get_var("density")
                # xmom = ccdata.get_var("x-momentum")
                # ymom = ccdata.get_var("y-momentum")
                ener = ccdata.get_var("energy")

                grav = ccdata.get_aux("grav")
                gamma = ccdata.get_aux("gamma")

                q = flx.cons_to_prim_wrapper(ccdata.data, gamma, ivars, myg)

                rho = q[:,:,ivars.irho]
                u = q[:,myg.jhi,ivars.iu]
                v = q[:,myg.jhi,ivars.iv]
                p = q[:,:,ivars.ip]

                dens_base = rho[:, myg.jhi]
                # ke_base = 0.5*(xmom[:, myg.jhi]**2 + ymom[:, myg.jhi]**2) / \
                    # dens[:, myg.jhi]

                # eint_base = (ener[:, myg.jhi] - ke_base)/dens[:, myg.jhi]
                # pres_base = eos.pres(gamma, dens_base, eint_base)
                pres_base = p[:, myg.jhi]

                # we are assuming that the density is constant in this
                # formulation of HSE, so the pressure comes simply from
                # differencing the HSE equation
                W = np.sqrt(1-u**2-v**2)
                for j in range(myg.jhi+1, myg.jhi+myg.ng+1):
                    pres_above = pres_base + grav*dens_base*myg.dy
                    # rhoe = eos.rhoe(gamma, pres_above)
                    rhoh = eos.rhoh_from_rho_p(gamma, dens_base, pres_above)

                    # ener[:, j] = rhoe + ke_base
                    ener[:, j] = rhoh*W**2 - pres_above - dens_base

                    pres_base = pres_above.copy()

            else:
                raise NotImplementedError("variable not defined")

        else:
            msg.fail("error: hse BC not supported for xlb or xrb")

    else:
        msg.fail("error: bc type %s not supported" % (bc_name))
