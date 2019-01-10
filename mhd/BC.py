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

import math
import numpy as np


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
            if variable in ["density", "x-momentum", "y-momentum", "ymom_src", "E_src", "fuel", "ash"]:
                v = ccdata.get_var(variable)
                j = myg.jlo-1
                while j >= 0:
                    v[:, j] = v[:, myg.jlo]
                    j -= 1

            elif variable == "energy":
                dens = ccdata.get_var("density")
                xmom = ccdata.get_var("x-momentum")
                ymom = ccdata.get_var("y-momentum")
                ener = ccdata.get_var("energy")

                grav = ccdata.get_aux("grav")
                gamma = ccdata.get_aux("gamma")

                dens_base = dens[:, myg.jlo]
                ke_base = 0.5*(xmom[:, myg.jlo]**2 + ymom[:, myg.jlo]**2) / \
                    dens[:, myg.jlo]

                eint_base = (ener[:, myg.jlo] - ke_base)/dens[:, myg.jlo]
                pres_base = eos.pres(gamma, dens_base, eint_base)

                # we are assuming that the density is constant in this
                # formulation of HSE, so the pressure comes simply from
                # differencing the HSE equation
                j = myg.jlo-1
                while j >= 0:
                    pres_below = pres_base - grav*dens_base*myg.dy
                    rhoe = eos.rhoe(gamma, pres_below)

                    ener[:, j] = rhoe + ke_base

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
                dens = ccdata.get_var("density")
                xmom = ccdata.get_var("x-momentum")
                ymom = ccdata.get_var("y-momentum")
                ener = ccdata.get_var("energy")

                grav = ccdata.get_aux("grav")
                gamma = ccdata.get_aux("gamma")

                dens_base = dens[:, myg.jhi]
                ke_base = 0.5*(xmom[:, myg.jhi]**2 + ymom[:, myg.jhi]**2) / \
                    dens[:, myg.jhi]

                eint_base = (ener[:, myg.jhi] - ke_base)/dens[:, myg.jhi]
                pres_base = eos.pres(gamma, dens_base, eint_base)

                # we are assuming that the density is constant in this
                # formulation of HSE, so the pressure comes simply from
                # differencing the HSE equation
                for j in range(myg.jhi+1, myg.jhi+myg.ng+1):
                    pres_above = pres_base + grav*dens_base*myg.dy
                    rhoe = eos.rhoe(gamma, pres_above)

                    ener[:, j] = rhoe + ke_base

                    pres_base = pres_above.copy()

            else:
                raise NotImplementedError("variable not defined")

        else:
            msg.fail("error: hse BC not supported for xlb or xrb")

    elif bc_name == "ramp":
        # Boundary conditions for double Mach reflection problem

        gamma = ccdata.get_aux("gamma")

        if bc_edge == "xlb":
            # lower x boundary
            # inflow condition with post shock setup

            v = ccdata.get_var(variable)
            i = myg.ilo - 1
            if variable in ["density", "x-momentum", "y-momentum", "energy"]:
                val = inflow_post_bc(variable, gamma)
                while i >= 0:
                    v[i, :] = val
                    i = i - 1
            else:
                v[:, :] = 0.0   # no source term

        elif bc_edge == "ylb":
            # lower y boundary
            # for x > 1./6., reflective boundary
            # for x < 1./6., inflow with post shock setup

            if variable in ["density", "x-momentum", "y-momentum", "energy"]:
                v = ccdata.get_var(variable)
                j = myg.jlo - 1
                jj = 0
                while j >= 0:
                    xcen_l = myg.x < 1.0/6.0
                    xcen_r = myg.x >= 1.0/6.0
                    v[xcen_l, j] = inflow_post_bc(variable, gamma)

                    if variable == "y-momentum":
                        v[xcen_r, j] = -1.0*v[xcen_r, myg.jlo+jj]
                    else:
                        v[xcen_r, j] = v[xcen_r, myg.jlo+jj]
                    j = j - 1
                    jj = jj + 1
            else:
                v = ccdata.get_var(variable)
                v[:, :] = 0.0   # no source term

        elif bc_edge == "yrb":
            # upper y boundary
            # time-dependent boundary, the shockfront moves with a 10 mach velocity forming an angle
            # to the x-axis of 30 degrees clockwise.
            # x coordinate of the grid is used to judge whether the cell belongs to pure post shock area,
            # the pure pre shock area or the mixed area.

            if variable in ["density", "x-momentum", "y-momentum", "energy"]:
                v = ccdata.get_var(variable)
                for j in range(myg.jhi+1, myg.jhi+myg.ng+1):
                    shockfront_up = 1.0/6.0 + (myg.y[j] + 0.5*myg.dy*math.sqrt(3))/math.tan(math.pi/3.0) \
                                    + (10.0/math.sin(math.pi/3.0))*ccdata.t
                    shockfront_down = 1.0/6.0 + (myg.y[j] - 0.5*myg.dy*math.sqrt(3))/math.tan(math.pi/3.0) \
                                    + (10.0/math.sin(math.pi/3.0))*ccdata.t
                    shockfront = np.array([shockfront_down, shockfront_up])
                    for i in range(myg.ihi+myg.ng+1):
                        v[i, j] = 0.0
                        cx_down = myg.x[i] - 0.5*myg.dx*math.sqrt(3)
                        cx_up = myg.x[i] + 0.5*myg.dx*math.sqrt(3)
                        cx = np.array([cx_down, cx_up])

                        for sf in shockfront:
                            for x in cx:
                                if x < sf:
                                    v[i, j] = v[i, j] + 0.25*inflow_post_bc(variable, gamma)
                                else:
                                    v[i, j] = v[i, j] + 0.25*inflow_pre_bc(variable, gamma)
            else:
                v = ccdata.get_var(variable)
                v[:, :] = 0.0   # no source term

    else:
        msg.fail("error: bc type %s not supported" % (bc_name))


def inflow_post_bc(var, g):
    # inflow boundary condition with post shock setup
    r_l = 8.0
    u_l = 7.1447096
    v_l = -4.125
    p_l = 116.5
    if var == "density":
        vl = r_l
    elif var == "x-momentum":
        vl = r_l*u_l
    elif var == "y-momentum":
        vl = r_l*v_l
    elif var == "energy":
        vl = p_l/(g - 1.0) + 0.5*r_l*(u_l*u_l + v_l*v_l)
    else:
        vl = 0.0
    return vl


def inflow_pre_bc(var, g):
    # pre shock setup
    r_r = 1.4
    u_r = 0.0
    v_r = 0.0
    p_r = 1.0
    if var == "density":
        vl = r_r
    elif var == "x-momentum":
        vl = r_r*u_r
    elif var == "y-momentum":
        vl = r_r*v_r
    elif var == "energy":
        vl = p_r/(g - 1.0) + 0.5*r_r*(u_r*u_r + v_r*v_r)
    else:
        vl = 0.0
    return vl
