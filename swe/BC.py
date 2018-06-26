"""
swe-specific boundary conditions.  Here, in particular, we
implement an HSE BC in the vertical direction.

Note: the pyro BC routines operate on a single variable at a time, so
some work will necessarily be repeated.

Also note: we may come in here with the aux_data (source terms), so
we'll do a special case for them

"""

import math
import numpy as np

from util import msg


def user(bc_name, bc_edge, variable, ccdata):
    """
    A hydrostatic boundary.  This integrates the equation of HSE into
    the ghost cells to get the pressure and height under the assumption
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
    variable : {'height', 'x-momentum', 'y-momentum', 'energy'}
        The variable whose ghost cells we are filling
    ccdata : CellCenterData2d object
        The data object

    """

    myg = ccdata.grid

    if bc_name == "hse":

        if bc_edge == "ylb":

            # lower y boundary

            # we will take the height to be constant, the velocity to
            # be outflow, and the pressure to be in HSE
            if variable in ["height", "x-momentum", "y-momentum", "fuel", "ash"]:
                v = ccdata.get_var(variable)
                j = myg.jlo-1
                while j >= 0:
                    v[:, j] = v[:, myg.jlo]
                    j -= 1
            else:
                raise NotImplementedError("variable not defined")

        elif bc_edge == "yrb":

            # upper y boundary

            # we will take the height to be constant, the velocity to
            # be outflow, and the pressure to be in HSE
            if variable in ["height", "x-momentum", "y-momentum", "fuel", "ash"]:
                v = ccdata.get_var(variable)
                for j in range(myg.jhi+1, myg.jhi+myg.ng+1):
                    v[:, j] = v[:, myg.jhi]
            else:
                raise NotImplementedError("variable not defined")

        else:
            msg.fail("error: hse BC not supported for xlb or xrb")

    elif bc_name == "ramp":
        # Boundary conditions for double Mach reflection problem

        g = ccdata.get_aux("g")

        if bc_edge == "xlb":
            # lower x boundary
            # inflow condition with post shock setup

            v = ccdata.get_var(variable)
            i = myg.ilo - 1
            if variable in ["height", "x-momentum", "y-momentum"]:
                val = inflow_post_bc(variable, g)
                while i >= 0:
                    v[i, :] = val
                    i = i - 1
            else:
                v[:, :] = 0.0   # no source term

        elif bc_edge == "ylb":
            # lower y boundary
            # for x > 1./6., reflective boundary
            # for x < 1./6., inflow with post shock setup

            if variable in ["height", "x-momentum", "y-momentum"]:
                v = ccdata.get_var(variable)
                j = myg.jlo - 1
                jj = 0
                while j >= 0:
                    xcen_l = myg.x < 1.0/6.0
                    xcen_r = myg.x >= 1.0/6.0
                    v[xcen_l, j] = inflow_post_bc(variable, g)

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

            if variable in ["height", "x-momentum", "y-momentum"]:
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
                                    v[i, j] = v[i, j] + 0.25*inflow_post_bc(variable, g)
                                else:
                                    v[i, j] = v[i, j] + 0.25*inflow_pre_bc(variable, g)
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
    if var == "height":
        vl = r_l
    elif var == "x-momentum":
        vl = r_l*u_l
    elif var == "y-momentum":
        vl = r_l*v_l
    else:
        vl = 0.0
    return vl


def inflow_pre_bc(var, g):
    # pre shock setup
    r_r = 1.4
    u_r = 0.0
    v_r = 0.0
    if var == "height":
        vl = r_r
    elif var == "x-momentum":
        vl = r_r*u_r
    elif var == "y-momentum":
        vl = r_r*v_r
    else:
        vl = 0.0
    return vl
