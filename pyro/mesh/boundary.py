"""
Methods to manage boundary conditions
"""


from pyro.util import msg

# keep track of whether the BCs are solid walls (passed into the
# Riemann solver).
bc_solid = {}
bc_solid["outflow"] = False
bc_solid["periodic"] = False
bc_solid["reflect"] = True
bc_solid["reflect-even"] = True
bc_solid["reflect-odd"] = True
bc_solid["dirichlet"] = True
bc_solid["neumann"] = False

ext_bcs = {}


def define_bc(bc_type, function, is_solid=False):
    """
    use this to extend the types of boundary conditions supported
    on a solver-by-solver basis.  Here we pass in the reference to
    a function that can be called with the data that needs to be
    filled.  is_solid indicates whether it should be interpreted as
    a solid wall (no flux through the BC)"
    """

    bc_solid[bc_type] = is_solid
    ext_bcs[bc_type] = function


def _set_reflect(odd_reflect_dir, dir_string):
    if odd_reflect_dir == dir_string:
        return "reflect-odd"
    return "reflect-even"


class BCProp:
    """
    A simple container to hold properties of the boundary conditions.
    """

    def __init__(self, xl_prop, xr_prop, yl_prop, yr_prop):
        self.xl = xl_prop
        self.xr = xr_prop
        self.yl = yl_prop
        self.yr = yr_prop


def bc_is_solid(bc):
    """
    return a container class indicating which boundaries are solid walls
    """
    solid = BCProp(int(bc_solid[bc.xlb]),
                   int(bc_solid[bc.xrb]),
                   int(bc_solid[bc.ylb]),
                   int(bc_solid[bc.yrb]))
    return solid


class BC:
    """Boundary condition container -- hold the BCs on each boundary
    for a single variable.

    For Neumann and Dirichlet BCs, a function callback can be stored
    for inhomogeous BCs.  This function should provide the value on
    the physical boundary (not cell center).  This is evaluated on the
    relevant edge when the __init__ routine is called.  For this
    reason, you need to pass in a grid object.  Note: this only
    ensures that the first ghost cells is consistent with the BC
    value.

    """

    def __init__(self,
                 xlb="outflow", xrb="outflow",
                 ylb="outflow", yrb="outflow",
                 xl_func=None, xr_func=None,
                 yl_func=None, yr_func=None,
                 grid=None,
                 odd_reflect_dir=""):
        """
        Create the BC object.

        Parameters
        ----------
        xlb : {'outflow', 'periodic', 'reflect', 'reflect-even',
               'reflect-odd', 'dirichlet', 'neumann',
               user-defined}, optional
            The type of boundary condition to enforce on the lower
            x boundary.  user-defined requires one to have defined
            a new boundary condition type using define_bc()

        xrb : {'outflow', 'periodic', 'reflect', 'reflect-even',
               'reflect-odd', 'dirichlet', 'neumann',
               user-defined}, optional
            The type of boundary condition to enforce on the upper
            x boundary.  user-defined requires one to have defined
            a new boundary condition type using define_bc()

        ylb : {'outflow', 'periodic', 'reflect', 'reflect-even',
               'reflect-odd', 'dirichlet', 'neumann',
               user-defined}, optional
            The type of boundary condition to enforce on the lower
            y boundary.  user-defined requires one to have defined
            a new boundary condition type using define_bc()

        yrb : {'outflow', 'periodic', 'reflect', 'reflect-even',
               'reflect-odd', 'dirichlet', 'neumann',
               user-defined}, optional
            The type of boundary condition to enforce on the upper
            y boundary.  user-defined requires one to have defined
            a new boundary condition type using define_bc()

        odd_reflect_dir : {'x', 'y'}, optional
            The direction along which reflection should be odd
            (sign changes).  If not specified, a boundary condition
            of 'reflect' will always be set to 'reflect-even'

        xl_func : function, optional
            A function, f(y), that provides the value of the
            Dirichlet or Neumann BC on the -x physical boundary.

        xr_func : function, optional
            A function, f(y), that provides the value of the
            Dirichlet or Neumann BC on the +x physical boundary.

        yl_func : function, optional
            A function, f(x), that provides the value of the
            Dirichlet or Neumann BC on the -y physical boundary.

        yr_func : function, optional
            A function, f(x), that provides the value of the
            Dirichlet or Neumann BC on the +y physical boundary.

        grid : a Grid2d object, optional
            The grid object is used for evaluating the function
            to define the boundary values for inhomogeneous
            Dirichlet and Neumann BCs.  It is required if
            any functions are passed in.
        """

        # note: "reflect" is ambiguous and will be converted into
        # either reflect-even (the default) or reflect-odd if
        # odd_reflect_dir specifies the corresponding direction ("x",
        # "y")

        valid = list(bc_solid.keys())

        # -x boundary
        if xlb in valid:
            self.xlb = xlb
            if self.xlb == "reflect":
                self.xlb = _set_reflect(odd_reflect_dir, "x")
        else:
            msg.fail(f"ERROR: xlb = {xlb} invalid BC")

        # +x boundary
        if xrb in valid:
            self.xrb = xrb
            if self.xrb == "reflect":
                self.xrb = _set_reflect(odd_reflect_dir, "x")
        else:
            msg.fail(f"ERROR: xrb = {xrb} invalid BC")

        # -y boundary
        if ylb in valid:
            self.ylb = ylb
            if self.ylb == "reflect":
                self.ylb = _set_reflect(odd_reflect_dir, "y")
        else:
            msg.fail(f"ERROR: ylb = {ylb} invalid BC")

        # +y boundary
        if yrb in valid:
            self.yrb = yrb
            if self.yrb == "reflect":
                self.yrb = _set_reflect(odd_reflect_dir, "y")
        else:
            msg.fail(f"ERROR: yrb = {yrb} invalid BC")

        # periodic checks
        if ((xlb == "periodic" and xrb != "periodic") or
                (xrb == "periodic" and xlb != "periodic")):
            msg.fail("ERROR: both xlb and xrb must be periodic")

        if ((ylb == "periodic" and yrb != "periodic") or
                (yrb == "periodic" and ylb != "periodic")):
            msg.fail("ERROR: both ylb and yrb must be periodic")

        # inhomogeneous functions for Dirichlet or Neumann
        self.xl_value = self.xr_value = self.yl_value = self.yr_value = None

        if xl_func is not None:
            self.xl_value = xl_func(grid.y)
        if xr_func is not None:
            self.xr_value = xr_func(grid.y)
        if yl_func is not None:
            self.yl_value = yl_func(grid.x)
        if yr_func is not None:
            self.yr_value = yr_func(grid.x)

    def __str__(self):
        """ print out some basic information about the BC object """

        string = f"BCs: -x: {self.xlb}  +x: {self.xrb}  -y: {self.ylb}  +y: {self.yrb}"

        return string
