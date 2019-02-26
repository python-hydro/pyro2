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
#
# import h5py
#
from util import msg
#
# import mesh.boundary as bnd
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
                 area_func=None, h_func=None, R_func=None):
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

        self.kappa, self.gamma_fcx, self.gamma_fcy = self.calculate_metric_elements(
            area_func, h_func)

        self.R_fcx, self.R_fcy = self.calculate_rotation_matrices(R_func)

        # self.R_fcx = self.scratch_array() + 1.0
        # self.R_fcy = self.scratch_array() + 1.0
        # self.R_T_fcx = self.scratch_array() + 1.0
        # self.R_T_fcy = self.scratch_array() + 1.0

    def calculate_metric_elements(self, area, h):
        """
        Given the functions for the area and line elements, calculate them on
        the grid.
        """

        kappa = self.scratch_array()

        kappa[:, :] = area(self) / (self.dx * self.dy)

        hx = self.scratch_array()
        hy = self.scratch_array()

        hx[:, :] = h(1, self)
        hy[:, :] = h(2, self)

        return kappa, hx / self.dy, hy / self.dx

    def calculate_rotation_matrices(self, R):
        """
        We're going to use partial functions here as the grid knows nothing of
        the variables.
        """

        R_fcx = partial(R, 1, self)
        R_fcy = partial(R, 2, self)

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
