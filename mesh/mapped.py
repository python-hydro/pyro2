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
from sympy.abc import x, y, z
from random import random
from numpy.testing import assert_array_almost_equal
from numba import njit

from util import msg
import mesh.boundary as bnd
import mesh.array_indexer as ai
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

    def __init__(self, map_func, nx, ny, ng=1,
                 xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0):
        """
        Create a MappedGrid2d object.

        The only data that we require is the number of points that
        make up the mesh in each direction.  Optionally we take the
        extrema of the domain (default is [0,1]x[0,1]) and number of
        ghost cells (default is 1).

        Note that the Grid2d object only defines the discretization,
        it does not know about the boundary conditions, as these can
        vary depending on the variable.

        Parameters
        ----------
        map_func : sympy.Matrix
        nx : int
            Number of zones in the x-direction
        ny : int
            Number of zones in the y-direction
        ng : int, optional
            Number of ghost cells
        xmin : float, optional
            Mapped coordinate at the lower x boundary
        xmax : float, optional
            Mapped coordinate at the upper x boundary
        ymin : float, optional
            Mapped coordinate at the lower y boundary
        ymax : float, optional
            Mapped coordinate at the upper y boundary
        """

        super().__init__(nx, ny, ng, xmin, xmax, ymin, ymax)

        # we need to add a z-direction so that we can calculate the cross product
        # of the basis vectors
        self.map = map_func(self).col_join(sympy.Matrix([z]))

        print("Calculating scaling factors...")
        self.kappa, self.gamma_fcx, self.gamma_fcy = self.calculate_metric_elements()

        print("Calculating rotation matrices...")
        self.R_fcx, self.R_fcy = self.calculate_rotation_matrices()

    @staticmethod
    def norm(z):
        return sympy.sqrt(z.dot(z))

    def sym_rotation_matrix(self):
        """
        Use sympy to calculate the rotation matrices
        """

        Rx = sympy.zeros(2)
        Ry = sympy.zeros(2)

        Rx[0, 0] = sympy.simplify(self.map[1].diff(y))
        Rx[0, 1] = sympy.simplify(self.map[0].diff(y))
        Rx[1, 0] = -sympy.simplify(self.map[0].diff(y))
        Rx[1, 1] = sympy.simplify(self.map[1].diff(y))

        Ry[0, 0] = sympy.simplify(self.map[0].diff(x))
        Ry[0, 1] = sympy.simplify(self.map[1].diff(x))
        Ry[1, 0] = -sympy.simplify(self.map[1].diff(x))
        Ry[1, 1] = sympy.simplify(self.map[0].diff(x))

        # normalize
        Rx[0, :] /= self.norm(Rx[0, :])
        Rx[1, :] /= self.norm(Rx[1, :])
        Ry[0, :] /= self.norm(Ry[0, :])
        Ry[1, :] /= self.norm(Ry[1, :])

        Rx = sympy.simplify(Rx)
        Ry = sympy.simplify(Ry)

        # check rotation matrices - do this by substituting in random (non-zero)
        # numbers as sympy is not great at cancelling things
        assert_array_almost_equal((Rx @ Rx.T).subs(
            {x: random() + 0.01, y: random() + 0.01}), np.eye(2))
        assert_array_almost_equal((Ry @ Ry.T).subs(
            {x: random() + 0.01, y: random() + 0.01}), np.eye(2))

        return sympy.simplify(Rx), sympy.simplify(Ry)

    def calculate_metric_elements(self):
        """
        Given the functions for the area and line elements, calculate them on
        the grid.
        """

        kappa = self.scratch_array()
        hx = self.scratch_array()
        hy = self.scratch_array()

        # calculate physical coordinates of corners
        c1 = self.physical_coords(self.xl, self.yl)
        c2 = self.physical_coords(self.xl, self.yr)
        c3 = self.physical_coords(self.xr, self.yr)
        c4 = self.physical_coords(self.xr, self.yl)

        def mapped_distance(m1, m2):
            return np.sqrt((m1[0] - m2[0])**2 + (m1[1] - m2[1])**2)

        def mapped_area(i,j):
            # find vectors of diagonals (and pad out z-direction with a zero)
            p = np.append(c3[:,i,j] - c1[:,i,j], 0)
            q = np.append(c4[:,i,j] - c2[:,i,j], 0)

            # area is half the cross product
            return 0.5 * np.abs(np.cross(p, q)[-1])

        for i in range(self.qx):
            for j in range(self.qy):
                hx[i, j] = mapped_distance(c1[:,i,j], c2[:,i,j])
                hy[i, j] = mapped_distance(c1[:,i,j], c4[:,i,j])
                kappa[i, j] = mapped_area(i, j)

        return kappa / (self.dx * self.dy), hx / self.dy, hy / self.dx

    def calculate_rotation_matrices(self):
        """
        Calculate the rotation matrices on the cell interfaces.
        It will return this as functions of nvar, ixmom and iymom - the grid
        itself knows nothing of the variables, so these must be specified
        by the MappedCellCenterData2d object.
        """

        # if isinstance(self.map, sympy.Matrix):
        sym_Rx, sym_Ry = self.sym_rotation_matrix()
        print('Rx = ', sym_Rx)
        print('Ry = ', sym_Ry)

        Rx = sympy.lambdify((x, y), sym_Rx, modules="sympy")
        Ry = sympy.lambdify((x, y), sym_Ry, modules="sympy")

        def R_fcx(nvar, ixmom, iymom):
            R_fc = self.scratch_array(nvar=(nvar, nvar))

            R_mat = np.eye(nvar)

            for i in range(self.qx):
                for j in range(self.qy):
                    R_fc[i, j, :, :] = R_mat

                    R_fc[i, j, ixmom:iymom + 1, ixmom:iymom +
                         1] = np.array(Rx(self.xl[i], self.y[j]))

            return R_fc

        def R_fcy(nvar, ixmom, iymom):
            R_fc = self.scratch_array(nvar=(nvar, nvar))

            R_mat = np.eye(nvar)

            for i in range(self.qx):
                for j in range(self.qy):
                    R_fc[i, j, :, :] = R_mat

                    R_fc[i, j, ixmom:iymom + 1, ixmom:iymom +
                         1] = np.array(Ry(self.x[i], self.yl[j]))
            return R_fc

        return R_fcx, R_fcy

    def scratch_array(self, nvar=1):
        """
        return a standard numpy array dimensioned to have the size
        and number of ghostcells as the parent grid.

        Here I've generalized the version in Grid2d so that we can define
        tensors (not just scalars and vectors) e.g. the rotation matrices
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

    def physical_coords(self, xx=None, yy=None):

        if xx is None:
            xs = self.x2d
        elif not np.isscalar(xx) and len(xx.shape) == 1:
            xs = np.repeat(xx, len(yy))
            xs.shape = (len(xx), len(yy))
        else:
            xs = xx

        if yy is None:
            ys = self.y2d
        elif not np.isscalar(yy) and len(yy.shape) == 1:
            ys = np.repeat(yy, len(xx))
            ys.shape = (len(yy), len(xx))
            ys = np.transpose(ys)
        else:
            ys = yy

        xs_t = sympy.lambdify((x, y), self.map[0])
        ys_t = sympy.lambdify((x, y), self.map[1])

        return np.array([xs_t(xs, ys), ys_t(xs, ys)])


class MappedCellCenterData2d(CellCenterData2d):

    def __init__(self, grid, dtype=np.float64):

        super().__init__(grid, dtype=dtype)

        self.R_fcx = []
        self.R_fcy = []

    def make_rotation_matrices(self, ivars):
        """
        The grid knows nothing of the variables, so we're going to define
        the actual rotation matrices here by passing in the variable data
        to the rotation matrix function.
        """

        self.R_fcx = self.grid.R_fcx(ivars.nvar, ivars.ixmom, ivars.iymom)
        self.R_fcy = self.grid.R_fcy(ivars.nvar, ivars.ixmom, ivars.iymom)

    def fill_BC(self, name):
        n = self.names.index(name)
        self.fill_mapped_ghost(name, n=n, bc=self.BCs[name])

        # that will handle the standard type of BCs, but if we asked
        # for a custom BC, we handle it here
        if self.BCs[name].xlb in bnd.ext_bcs.keys():
            bnd.ext_bcs[self.BCs[name].xlb](self.BCs[name].xlb, "xlb", name, self)
        if self.BCs[name].xrb in bnd.ext_bcs.keys():
            bnd.ext_bcs[self.BCs[name].xrb](self.BCs[name].xrb, "xrb", name, self)
        if self.BCs[name].ylb in bnd.ext_bcs.keys():
            bnd.ext_bcs[self.BCs[name].ylb](self.BCs[name].ylb, "ylb", name, self)
        if self.BCs[name].yrb in bnd.ext_bcs.keys():
            bnd.ext_bcs[self.BCs[name].yrb](self.BCs[name].yrb, "yrb", name, self)

    def fill_mapped_ghost(self, name, n=0, bc=None):
        """
        This replaces the fill_ghost routine in array_indexer to add some
        scaling factors for mapped grids.

        We'll call fill_ghost then go back and fix the reflect-odd boundaries.

        It would be great if we could pass the variables object in here as we
        actually only need to do this rotation if n == ixmom or iymom.
        """
        myd = self.data

        myd.fill_ghost(n=n, bc=bc)

        if name == 'x-momentum' or name == 'y-momentum':

            # -x boundary
            if bc.xlb in ["reflect-odd", "dirichlet"]:
                if bc.xl_value is None:
                    for i in range(myd.g.ilo):
                        for j in range(myd.g.qy):
                            q_rot = self.R_fcy[myd.g.ilo, j] @ myd[2*myd.g.ng-i-1, j, :]
                            q_rot[n] = -q_rot[n]
                            myd[i, j, n] = (self.R_fcy[myd.g.ilo, j].T @ q_rot)[n]
                else:
                    for j in range(myd.g.qy):
                        q_rot = self.R_fcy[myd.g.ilo, j] @ myd[myd.g.ilo, j, :]
                        q_rot[n] = 2*bc.xl_value - q_rot[n]
                        myd[myd.g.ilo-1, j, n] = (self.R_fcy[myd.g.ilo, j].T @ q_rot)[n]

            # +x boundary
            if bc.xrb in ["reflect-odd", "dirichlet"]:
                if bc.xr_value is None:
                    for i in range(myd.g.ng):
                        for j in range(myd.g.qy):
                            i_bnd = myd.g.ihi+1+i
                            i_src = myd.g.ihi-i

                            q_rot = self.R_fcy[myd.g.ihi+1, j] @ myd[i_src, j, :]
                            q_rot[n] = -q_rot[n]

                            myd[i_bnd, j, n] = (self.R_fcy[myd.g.ihi+1, j].T @ q_rot)[n]
                else:
                    for j in range(myd.g.qy):
                        q_rot = self.R_fcy[myd.g.ihi+1, j] @ myd[myd.g.ihi, j, :]
                        q_rot[n] = 2*bc.xr_value - q_rot[n]

                        myd[myd.g.ihi+1, j, n] = (self.R_fcy[myd.g.ihi+1, j].T @ q_rot)[n]

            # -y boundary
            if bc.ylb in ["reflect-odd", "dirichlet"]:
                if bc.yl_value is None:
                    for i in range(myd.g.qx):
                        for j in range(myd.g.jlo):
                            q_rot = self.R_fcx[i, myd.g.jlo] @ myd[i, 2*myd.g.ng-j-1, :]
                            q_rot[n] = -q_rot[n]
                            myd[i, j, n] = (self.R_fcx[i, myd.g.jlo].T @ q_rot)[n]
                else:
                    for i in range(myd.g.qx):
                        q_rot = self.R_fcx[i, myd.g.jlo] @ myd[i, myd.g.jlo, :]
                        q_rot[n] = 2*bc.yl_value - q_rot[n]
                        myd[i, myd.g.jlo-1, n] = (self.R_fcx[i, myd.g.jlo].T @ q_rot)[n]

            # +y boundary
            if bc.yrb in ["reflect-odd", "dirichlet"]:
                if bc.yr_value is None:
                    for i in range(myd.g.qx):
                        for j in range(myd.g.ng):
                            j_bnd = myd.g.jhi+1+j
                            j_src = myd.g.jhi-j

                            q_rot = self.R_fcx[i, myd.g.jhi+1] @ myd[i, j_src, n]
                            q_rot[n] = -q_rot[n]

                            myd[i, j_bnd, n] = (self.R_fcx[i, myd.g.jhi+1].T @ q_rot)[n]
                else:
                    for i in range(myd.g.qx):

                        q_rot = self.R_fcx[i, myd.g.jhi+1] @ myd[i, myd.g.jhi, n]
                        q_rot[n] = 2*bc.yr_value - q_rot[n]

                        myd[:, myd.g.jhi+1, n] = (self.R_fcx[i, myd.g.jhi+1].T @ q_rot)[n]


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
