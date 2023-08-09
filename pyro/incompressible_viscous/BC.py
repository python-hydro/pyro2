"""
Solver-specific boundary conditions.  Here, in particular, we
implement a "moving-lid" option, i.e. fixed tangential velocity at a wall.
"""

from pyro.util import msg


def user(bc_name, bc_edge, variable, ccdata):
    """
    A moving lid boundary. This fixes the tangent velocity at the boundary
    to 1, and the y-velocity to 0.

    Parameters
    ----------
    bc_name : {'moving_lid'}
        The descriptive name for the boundary condition -- this allows
        for pyro to have multiple types of user-supplied boundary
        conditions.  For this module, it needs to be 'moving_lid'.
    bc_edge : {'ylb', 'yrb', 'xlb', 'xrb}
        The boundary to update: ylb = lower y; yrb = upper y.
        xlb = left x; xrb = right x
    ccdata : CellCenterData2d object
        The data object
    """

    myg = ccdata.grid

    if bc_name == "moving_lid":

        if bc_edge == "yrb":

            if variable in ("x-velocity", "u"):
                v = ccdata.get_var(variable)
                for j in range(myg.jhi+1, myg.jhi+myg.ng+1):
                    v[:, j] = 1.0  # unit velocity

            elif variable in ("y-velocity", "v"):
                v = ccdata.get_var(variable)
                for j in range(myg.jhi+1, myg.jhi+myg.ng+1):
                    v[:, j] = 0.0

            else:
                raise NotImplementedError("variable not defined")

        else:
            msg.fail("error: moving_lid BC only implemented for 'yrb' (top boundary)")

    else:
        msg.fail(f"error: bc type {bc_name} not supported")
