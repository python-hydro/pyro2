"""
The patch module defines the classes necessary to describe finite-volume
data and the grid that it lives on.

Typical usage:

* create the grid::

   grid = Grid2d(nx, ny)

* create the data that lives on that grid::

   data = CellCenterData2d(grid)

   bc = BC(xlb="reflect", xrb="reflect",
          ylb="outflow", yrb="outflow")
   data.register_var("density", bc)
   ...

   data.create()

* initialize some data::

   dens = data.get_var("density")
   dens[:, :] = ...


* fill the ghost cells::

   data.fill_BC("density")

"""
from __future__ import print_function

import numpy as np
import sympy
from sympy.abc import x, y
from util import msg
import mesh.array_indexer as ai
from functools import partial
from mesh.patch import Grid2d, CellCenterData2d


class MappedGrid2d(Grid2d):
    """
    the mapped 2-d grid class.  The grid object will contain the coordinate
    information (at various centerings).

    A basic (1-d) representation of the layout is::

       |     |      |     X     |     |      |     |     X     |      |     |
       +--*--+- // -+--*--X--*--+--*--+- // -+--*--+--*--X--*--+- // -+--*--+
          0          ng-1    ng   ng+1         ... ng+nx-1 ng+nx      2ng+nx-1

                            ilo                      ihi

       |<- ng guardcells->|<---- nx interior zones ----->|<- ng guardcells->|

    The '*' marks the data locations.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, nx, ny, ng=1,
                 xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
                 area_func=None, h_func=None, R_func=None, map_func=None):
        """
        Create a Grid2d object.

        The only data that we require is the number of points that
        make up the mesh in each direction.  Optionally we take the
        extrema of the domain (default is [0,1]x[0,1]) and number of
        ghost cells (default is 1).

        Note that the Grid2d object only defines the discretization,
        it does not know about the boundary conditions, as these can
        vary depending on the variable.

        Parameters
        ----------
        nx : int
            Number of zones in the x-direction
        ny : int
            Number of zones in the y-direction
        ng : int, optional
            Number of ghost cells
        xmin : float, optional
            Physical coordinate at the lower x boundary
        xmax : float, optional
            Physical coordinate at the upper x boundary
        ymin : float, optional
            Physical coordinate at the lower y boundary
        ymax : float, optional
            Physical coordinate at the upper y boundary
        """

        super().__init__(nx, ny, ng, xmin, xmax, ymin, ymax)

        self.map = map_func(self)

        self.kappa, self.gamma_fcx, self.gamma_fcy = self.calculate_metric_elements(
            area_func, h_func)

        self.R_fcx, self.R_fcy = self.calculate_rotation_matrices(R_func)

    @staticmethod
    def norm(z):
        return sympy.sqrt(z.dot(z))

    def sym_area_element(self):
        """
        Use sympy to calculate area element using https://mzucker.github.io/2018/04/12/sympy-part-3-moar-derivatives.html
        """

        def norm(z):
            return sympy.sqrt(z.dot(z))

        p1 = sympy.simplify(self.map.diff(x))
        p2 = sympy.simplify(self.map.diff(y))

        p1_cross_p2 = sympy.simplify(p1.cross(p2))

        dA = sympy.simplify(self.norm(p1_cross_p2))

        return dA

    def sym_line_elements(self):
        """
        Use sympy to calculate the line elements
        """

        l1 = sympy.simplify(sympy.Abs(self.map.diff(y)))
        l2 = sympy.simplify(sympy.Abs(self.map.diff(x)))

        return l1, l2

    def sym_rotation_matrix(self):
        """
        Use sympy to calculate the rotation matrix (which is just the Jacobian??)
        """

        J = sympy.Matrix(self.map[:-1]).jacobian((x, y))
        J[0,:] /= J[0,:].norm()
        J[1,:] /= J[1,:].norm()

        return J

    def calculate_metric_elements(self, area, h):
        """
        Given the functions for the area and line elements, calculate them on
        the grid.
        """

        kappa = self.scratch_array()
        hx = self.scratch_array()
        hy = self.scratch_array()

        if isinstance(self.map, sympy.Matrix):
            # calculate sympy formula on grid
            sym_dA = self.sym_area_element()

            sym_hx, sym_hy = self.sym_line_elements()

            for i in range(self.qx):
                for j in range(self.qy):
                    kappa[i, j] = sym_dA.subs((x, self.x2d[i, j]), (y, self.y2d[i, j]))
                    hx[i, j] = sym_hx.subs((x, self.x2d[i, j] - 0.5 *
                                   self.dx), (y, self.y2d[i, j]))[1]
                    hy[i, j] = sym_hy.subs(
                        (x, self.x2d[i, j]), (y, self.y2d[i, j] - 0.5 * self.dy))[0]

        else:
            kappa[:, :] = area(self) / (self.dx * self.dy)
            hx[:, :] = h(1, self) / self.dy
            hy[:, :] = h(2, self) / self.dx

        return kappa, hx, hy

    def calculate_rotation_matrices(self, R):
        """
        We're going to use partial functions here as the grid knows nothing of
        the variables.
        """

        if isinstance(self.map, sympy.Matrix):
            sym_R = self.sym_rotation_matrix()
            R = sympy.lambdify((x, y), sym_R)

            def R_fcx(nvar, ixmom, iymom):
                R_fc = self.scratch_array(nvar=(nvar, nvar))

                R_mat = np.eye(nvar)

                for i in range(self.qx):
                    for j in range(self.qy):
                        R_fc[i, j, :, :] = R_mat

                        R_fc[i, j, ixmom:iymom + 1, ixmom:iymom +
                             1] = R(self.x2d[i, j] - 0.5 * self.dx, self.y2d[i, j])

                return R_fc

            # print(R_fcx(4, 2, 3))

            def R_fcy(nvar, ixmom, iymom):
                R_fc = self.scratch_array(nvar=(nvar, nvar))

                R_mat = np.eye(nvar)

                for i in range(self.qx):
                    for j in range(self.qy):
                        R_fc[i, j, :, :] = R_mat

                        R_fc[i, j, ixmom:iymom + 1, ixmom:iymom +
                             1] = R(self.x2d[i, j], self.y2d[i, j] - 0.5 * self.dy)
                return R_fc

            # print(R_fcy(4, 2, 3))
        else:
            R_fcx = partial(R, 1, self)
            R_fcy = partial(R, 2, self)

            # print(R_fcx(4, 2, 3))

        return R_fcx, R_fcy

    def scratch_array(self, nvar=1):
        """
        return a standard numpy array dimensioned to have the size
        and number of ghostcells as the parent grid
        """

        def flatten(t):
            if not isinstance(t, tuple):
                return (t, )
            elif len(t) == 0:
                return ()
            else:
                return flatten(t[0]) + flatten(t[1:])

        if nvar == 1:
            _tmp = np.zeros((self.qx, self.qy), dtype=np.float64)
        else:
            _tmp = np.zeros((self.qx, self.qy) +
                            flatten(nvar), dtype=np.float64)
        return ai.ArrayIndexer(d=_tmp, grid=self)


class MappedCellCenterData2d(CellCenterData2d):

    def __init__(self, grid, dtype=np.float64):

        super().__init__(grid, dtype=dtype)

        # self.R_fcx = []
        # self.R_fcy = []

    def make_rotation_matrices(self, ivars):
        """
        The grid knows nothing of the variables, so we're going to define
        the actual rotation matrices here by passing in the variable data
        to the rotation matrix function.
        """

        self.R_fcx = self.grid.R_fcx(ivars.nvar, ivars.ixmom, ivars.iymom)
        self.R_fcy = self.grid.R_fcy(ivars.nvar, ivars.ixmom, ivars.iymom)


def mapped_cell_center_data_clone(old):
    """
    Create a new CellCenterData2d object that is a copy of an existing
    one

    Parameters
    ----------
    old : CellCenterData2d object
        The CellCenterData2d object we wish to copy

    Note
    ----
    It may be that this whole thing can be replaced with a copy.deepcopy()

    """

    if not isinstance(old, MappedCellCenterData2d):
        msg.fail("Can't clone object")

    # we may be a type derived from CellCenterData2d, so use the same
    # type
    myt = type(old)
    new = myt(old.grid, dtype=old.dtype)

    for n in range(old.nvar):
        new.register_var(old.names[n], old.BCs[old.names[n]])

    new.create()

    new.aux = old.aux.copy()
    new.data = old.data.copy()
    new.derives = old.derives.copy()

    new.R_fcx = old.R_fcx.copy()
    new.R_fcy = old.R_fcy.copy()

    return new
