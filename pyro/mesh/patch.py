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

import h5py
import numpy as np

import pyro.mesh.boundary as bnd
from pyro.mesh.array_indexer import ArrayIndexer, ArrayIndexerFC
from pyro.util import msg


class Grid2d:
    """
    the 2-d grid class.  The grid object will contain the coordinate
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
                 xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0):
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

        # pylint: disable=too-many-arguments

        # size of grid
        self.nx = int(nx)
        self.ny = int(ny)
        self.ng = int(ng)

        self.qx = int(2*ng + nx)
        self.qy = int(2*ng + ny)

        # domain extrema
        self.xmin = xmin
        self.xmax = xmax

        self.ymin = ymin
        self.ymax = ymax

        # compute the indices of the block interior (excluding guardcells)
        self.ilo = self.ng
        self.ihi = self.ng + self.nx-1

        self.jlo = self.ng
        self.jhi = self.ng + self.ny-1

        # center of the grid (for convenience)
        self.ic = self.ilo + self.nx//2 - 1
        self.jc = self.jlo + self.ny//2 - 1

        # define the coordinate information at the left, center, and right
        # zone coordinates
        self.dx = (xmax - xmin)/nx

        self.xl = (np.arange(self.qx) - ng)*self.dx + xmin
        self.xr = (np.arange(self.qx) + 1.0 - ng)*self.dx + xmin
        self.x = 0.5*(self.xl + self.xr)

        self.dy = (ymax - ymin)/ny

        self.yl = (np.arange(self.qy) - ng)*self.dy + ymin
        self.yr = (np.arange(self.qy) + 1.0 - ng)*self.dy + ymin
        self.y = 0.5*(self.yl + self.yr)

        # 2-d versions of the zone coordinates (replace with meshgrid?)
        x2d = np.repeat(self.x, self.qy)
        x2d.shape = (self.qx, self.qy)
        self.x2d = x2d

        y2d = np.repeat(self.y, self.qx)
        y2d.shape = (self.qy, self.qx)
        y2d = np.transpose(y2d)
        self.y2d = y2d

    def scratch_array(self, nvar=1):
        """
        return a standard numpy array dimensioned to have the size
        and number of ghostcells as the parent grid
        """
        if nvar == 1:
            _tmp = np.zeros((self.qx, self.qy), dtype=np.float64)
        else:
            _tmp = np.zeros((self.qx, self.qy, nvar), dtype=np.float64)
        return ArrayIndexer(d=_tmp, grid=self)

    def coarse_like(self, N):
        """
        return a new grid object coarsened by a factor n, but with
        all the other properties the same
        """
        return Grid2d(self.nx//N, self.ny//N, ng=self.ng,
                      xmin=self.xmin, xmax=self.xmax,
                      ymin=self.ymin, ymax=self.ymax)

    def fine_like(self, N):
        """
        return a new grid object finer by a factor n, but with
        all the other properties the same
        """
        return Grid2d(self.nx*N, self.ny*N, ng=self.ng,
                      xmin=self.xmin, xmax=self.xmax,
                      ymin=self.ymin, ymax=self.ymax)

    def __str__(self):
        """ print out some basic information about the grid object """
        return f"2-d grid: nx = {self.nx}, ny = {self.ny}, ng = {self.ng}"

    def __eq__(self, other):
        """ are two grids equivalent? """
        result = (self.nx == other.nx and self.ny == other.ny and
                  self.ng == other.ng and
                  self.xmin == other.xmin and self.xmax == other.xmax and
                  self.ymin == other.ymin and self.ymax == other.ymax)

        return result


class CellCenterData2d:
    """
    A class to define cell-centered data that lives on a grid.  A
    CellCenterData2d object is built in a multi-step process before
    it can be used.

    * Create the object.  We pass in a grid object to describe where
      the data lives::

         my_data = patch.CellCenterData2d(myGrid)

    * Register any variables that we expect to live on this patch.
      Here BC describes the boundary conditions for that variable::

         my_data.register_var('density', BC)
         my_data.register_var('x-momentum', BC)
         ...

    * Register any auxillary data -- these are any parameters that are
      needed to interpret the data outside of the simulation (for
      example, the gamma for the equation of state)::

         my_data.set_aux(keyword, value)

    * Finish the initialization of the patch::

         my_data.create()

    This last step actually allocates the storage for the state
    variables.  Once this is done, the patch is considered to be
    locked.  New variables cannot be added.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, grid, dtype=np.float64):

        """
        Initialize the CellCenterData2d object.

        Parameters
        ----------
        grid : Grid2d object
            The grid upon which the data will live
        dtype : NumPy data type, optional
            The datatype of the data we wish to create (defaults to
            np.float64
        """

        self.grid = grid

        self.dtype = dtype
        self.data = None

        self.names = []
        self.vars = self.names  # backwards compatibility hack
        self.nvar = 0
        self.ivars = []

        self.aux = {}

        # derived variables will have a callback function
        self.derives = []

        self.BCs = {}

        # time
        self.t = -1.0

        self.initialized = 0

    def register_var(self, name, bc):
        """
        Register a variable with CellCenterData2d object.

        Parameters
        ----------
        name : str
            The variable name
        bc : BC object
            The boundary conditions that describe the actions to take
            for this variable at the physical domain boundaries.
        """

        if self.initialized == 1:
            msg.fail("ERROR: grid already initialized")

        self.names.append(name)
        self.nvar += 1

        self.BCs[name] = bc

    def set_aux(self, keyword, value):
        """
        Set any auxillary (scalar) data.  This data is simply carried
        along with the CellCenterData2d object

        Parameters
        ----------
        keyword : str
            The name of the datum
        value : any time
            The value to associate with the keyword
        """
        self.aux[keyword] = value

    def add_derived(self, func):
        """
        Register a function to compute derived variable

        Parameters
        ----------
        func : function
            A function to call to derive the variable.  This function
            should take two arguments, a CellCenterData2d object and a
            string variable name (or list of variables)
        """
        self.derives.append(func)

    def add_ivars(self, ivars):
        """
        Add ivars
        """

        self.ivars = ivars

    def create(self):
        """
        Called after all the variables are registered and allocates
        the storage for the state data.
        """

        if self.initialized == 1:
            msg.fail("ERROR: grid already initialized")

        _tmp = np.zeros((self.grid.qx, self.grid.qy, self.nvar),
                        dtype=self.dtype)
        self.data = ArrayIndexer(_tmp, grid=self.grid)

        self.initialized = 1

    def __str__(self):
        """ print out some basic information about the CellCenterData2d
            object """

        if self.initialized == 0:
            my_str = "CellCenterData2d object not yet initialized"
            return my_str

        my_str = f"cc data: nx = {self.grid.nx}, ny = {self.grid.ny}, ng = {self.grid.ng}\n"
        my_str += f"         nvars = {self.nvar}\n"
        my_str += "         variables:\n"

        for n in range(self.nvar):
            name = self.names[n]
            my_str += f"{name:>16s}: min: {self.min(name):15.10f}    max: {self.max(name):15.10f}\n"
            my_str += f"{' ':>16s}  BCs: -x: {self.BCs[name].xlb:12s} +x: {self.BCs[name].xrb:12s}"
            my_str += f" -y: {self.BCs[name].ylb:12s} +y: {self.BCs[name].yrb:12s}\n"

        return my_str

    def get_var(self, name):
        """
        Return a data array for the variable described by name.  Stored
        variables will be checked first, and then any derived variables
        will be checked.

        For a stored variable, changes made to this are automatically
        reflected in the CellCenterData2d object.

        Parameters
        ----------
        name : str
            The name of the variable to access

        Returns
        -------
        out : ndarray
            The array of data corresponding to the variable name

        """
        # ns = [self.names.index(name) for name in self.names]
        try:
            n = self.names.index(name)
        except ValueError:
            for f in self.derives:
                try:
                    var = f(self, name)
                except TypeError:
                    var = f(self, name, self.ivars, self.grid)
                if len(var) > 0:
                    return var
            raise KeyError(f"name {name} is not valid")
        else:
            return self.get_var_by_index(n)

    def get_var_by_index(self, n):
        """
        Return a data array for the variable with index n in the
        data array.  Any changes made to this are automatically
        reflected in the CellCenterData2d object.

        Parameters
        ----------
        n : int
            The index of the variable to access

        Returns
        -------
        out : ndarray
            The array of data corresponding to the index

        """
        return ArrayIndexer(d=self.data[:, :, n], grid=self.grid)

    def get_vars(self):
        """
        Return the entire data array.  Any changes made to this
        are automatically reflected in the CellCenterData2d object.

        Returns
        -------
        out : ndarray
            The array of data

        """
        return ArrayIndexer(d=self.data, grid=self.grid)

    def get_aux(self, keyword):
        """
        Get the auxillary data associated with keyword

        Parameters
        ----------
        keyword : str
            The name of the auxillary data to access

        Returns
        -------
        out : variable type
            The value corresponding to the keyword

        """
        if keyword in self.aux:
            return self.aux[keyword]

        return None

    def zero(self, name):
        """
        Zero out the data array associated with variable name.

        Parameters
        ----------
        name : str
            The name of the variable to zero

        """
        n = self.names.index(name)
        self.data[:, :, n] = 0.0

    def fill_BC_all(self):
        """
        Fill boundary conditions on all variables.
        """
        for name in self.names:
            self.fill_BC(name)

    def fill_BC(self, name):
        """
        Fill the boundary conditions.  This operates on a single state
        variable at a time, to allow for maximum flexibility.

        We do periodic, reflect-even, reflect-odd, and outflow

        Each variable name has a corresponding BC stored in the
        CellCenterData2d object -- we refer to this to figure out the
        action to take at each boundary.

        Parameters
        ----------
        name : str
            The name of the variable for which to fill the BCs.

        """

        n = self.names.index(name)
        self.data.fill_ghost(n=n, bc=self.BCs[name])

        # that will handle the standard type of BCs, but if we asked
        # for a custom BC, we handle it here
        if self.BCs[name].xlb in bnd.ext_bcs:
            try:
                bnd.ext_bcs[self.BCs[name].xlb](self.BCs[name].xlb, "xlb", name, self, self.ivars)
            except TypeError:
                bnd.ext_bcs[self.BCs[name].xlb](self.BCs[name].xlb, "xlb", name, self)
        if self.BCs[name].xrb in bnd.ext_bcs:
            try:
                bnd.ext_bcs[self.BCs[name].xrb](self.BCs[name].xrb, "xrb", name, self)
            except TypeError:
                bnd.ext_bcs[self.BCs[name].xrb](self.BCs[name].xrb, "xrb", name, self, self.ivars)
        if self.BCs[name].ylb in bnd.ext_bcs:
            try:
                bnd.ext_bcs[self.BCs[name].ylb](self.BCs[name].ylb, "ylb", name, self)
            except TypeError:
                bnd.ext_bcs[self.BCs[name].ylb](self.BCs[name].ylb, "ylb", name, self, self.ivars)
        if self.BCs[name].yrb in bnd.ext_bcs:
            try:
                bnd.ext_bcs[self.BCs[name].yrb](self.BCs[name].yrb, "yrb", name, self)
            except TypeError:
                bnd.ext_bcs[self.BCs[name].yrb](self.BCs[name].yrb, "yrb", name, self, self.ivars)

    def min(self, name, ng=0):
        """
        return the minimum of the variable name in the domain's valid region
        """
        n = self.names.index(name)
        return np.min(self.data.v(buf=ng, n=n))

    def max(self, name, ng=0):
        """
        return the maximum of the variable name in the domain's valid region
        """
        n = self.names.index(name)
        return np.max(self.data.v(buf=ng, n=n))

    def restrict(self, varname, N=2):
        """
        Restrict the variable varname to a coarser grid (factor of 2
        coarser) and return an array with the resulting data (and same
        number of ghostcells)
        """

        fine_grid = self.grid
        fdata = self.get_var(varname)

        # allocate an array for the coarsely gridded data
        coarse_grid = fine_grid.coarse_like(N)
        cdata = coarse_grid.scratch_array()

        # fill the coarse array with the restricted data -- just
        # by averaging the fine cells into the corresponding coarse cell
        # that encompasses them.
        if N == 2:
            cdata.v()[:, :] = \
                0.25*(fdata.v(s=2) + fdata.ip(1, s=2) +
                      fdata.jp(1, s=2) + fdata.ip_jp(1, 1, s=2))
        elif N == 4:
            cdata.v()[:, :] = \
                (fdata.v(s=4) +
                 fdata.ip(1, s=4) +
                 fdata.ip(2, s=4) + fdata.ip(3, s=4) +
                 fdata.jp(1, s=4) + fdata.ip_jp(1, 1, s=4) +
                 fdata.ip_jp(2, 1, s=4) + fdata.ip_jp(3, 1, s=4) +
                 fdata.jp(2, s=4) + fdata.ip_jp(1, 2, s=4) +
                 fdata.ip_jp(2, 2, s=4) + fdata.ip_jp(3, 2, s=4) +
                 fdata.jp(3, s=4) + fdata.ip_jp(1, 3, s=4) +
                 fdata.ip_jp(2, 3, s=4) + fdata.ip_jp(3, 3, s=4))/16.0

        else:
            raise ValueError("restriction is only allowed by 2 or 4")

        return cdata

    def prolong(self, varname):
        """
        Prolong the data in the current (coarse) grid to a finer
        (factor of 2 finer) grid.  Return an array with the resulting
        data (and same number of ghostcells).  Only the data for the
        variable varname will be operated upon.

        We will reconstruct the data in the zone from the
        zone-averaged variables using the same limited slopes as in
        the advection routine.  Getting a good multidimensional
        reconstruction polynomial is hard -- we want it to be bilinear
        and monotonic -- we settle for having each slope be
        independently monotonic::

                     (x)         (y)
           f(x,y) = m    x/dx + m    y/dy + <f>

        where the m's are the limited differences in each direction.
        When averaged over the parent cell, this reproduces <f>.

        Each zone's reconstrution will be averaged over 4 children::

           +-----------+     +-----+-----+
           |           |     |     |     |
           |           |     |  3  |  4  |
           |    <f>    | --> +-----+-----+
           |           |     |     |     |
           |           |     |  1  |  2  |
           +-----------+     +-----+-----+

        We will fill each of the finer resolution zones by filling all
        the 1's together, using a stride 2 into the fine array.  Then
        the 2's and ..., this allows us to operate in a vector
        fashion.  All operations will use the same slopes for their
        respective parents.

        """

        coarse_grid = self.grid
        cdata = self.get_var(varname)

        # allocate an array for the finely gridded data
        fine_grid = coarse_grid.fine_like(2)
        fdata = fine_grid.scratch_array()

        # slopes for the coarse data
        m_x = coarse_grid.scratch_array()
        m_x.v()[:, :] = 0.5*(cdata.ip(1) - cdata.ip(-1))

        m_y = coarse_grid.scratch_array()
        m_y.v()[:, :] = 0.5*(cdata.jp(1) - cdata.jp(-1))

        # fill the children
        fdata.v(s=2)[:, :] = cdata.v() - 0.25*m_x.v() - 0.25*m_y.v()      # 1 child
        fdata.ip(1, s=2)[:, :] = cdata.v() + 0.25*m_x.v() - 0.25*m_y.v()  # 2
        fdata.jp(1, s=2)[:, :] = cdata.v() - 0.25*m_x.v() + 0.25*m_y.v()  # 3
        fdata.ip_jp(1, 1, s=2)[:, :] = cdata.v() + 0.25*m_x.v() + 0.25*m_y.v()  # 4

        return fdata

    def write(self, filename):
        """
        create an output file in HDF5 format and write out our data and
        grid.
        """

        if not filename.endswith(".h5"):
            filename += ".h5"

        with h5py.File(filename, "w") as f:
            self.write_data(f)

    def write_data(self, f):
        """
        write the data out to an hdf5 file -- here, f is an h5py
        File pbject

        """

        # auxillary data
        gaux = f.create_group("aux")
        for k, v in self.aux.items():
            gaux.attrs[k] = v

        # grid information
        ggrid = f.create_group("grid")
        ggrid.attrs["nx"] = self.grid.nx
        ggrid.attrs["ny"] = self.grid.ny
        ggrid.attrs["ng"] = self.grid.ng

        ggrid.attrs["xmin"] = self.grid.xmin
        ggrid.attrs["xmax"] = self.grid.xmax
        ggrid.attrs["ymin"] = self.grid.ymin
        ggrid.attrs["ymax"] = self.grid.ymax

        # data
        gstate = f.create_group("state")

        for n in range(self.nvar):
            gvar = gstate.create_group(self.names[n])
            gvar.create_dataset("data",
                                data=self.get_var_by_index(n).v())
            gvar.attrs["xlb"] = self.BCs[self.names[n]].xlb
            gvar.attrs["xrb"] = self.BCs[self.names[n]].xrb
            gvar.attrs["ylb"] = self.BCs[self.names[n]].ylb
            gvar.attrs["yrb"] = self.BCs[self.names[n]].yrb

    def pretty_print(self, var, fmt=None):
        """print out the contents of the data array with pretty formatting
        indicating where ghost cells are."""
        a = self.get_var(var)
        a.pretty_print(fmt=fmt)


class FaceCenterData2d(CellCenterData2d):
    """
    A class to define face-centered data that lives on a grid.  Data
    can be face-centered in x or y.  This is built in the same multistep
    process  as a CellCenterData2d object"""

    def __init__(self, grid, idir, dtype=np.float64):
        """
        Initialize the FaceCenterData2d object

        Parameters
        ----------
        grid : Grid2d object
            The grid upon which the data will live
        idir : the direction in which we are face-centered (this will be
               1 for x or 2 for y)
        dtype : NumPy data type, optional
            The datatype of the data we wish to create (defaults to
            np.float64
        """

        super().__init__(grid, dtype=dtype)
        self.idir = idir

    def add_derived(self, func):
        raise NotImplementedError("derived variables not yet supported for face-centered data")

    def create(self):
        """Called after all the variables are registered and allocates the
        storage for the state data.  For face-centered data, we have
        one more zone in the face-centered direction.

        """

        if self.initialized == 1:
            msg.fail("ERROR: grid already initialized")

        if self.idir == 1:
            _tmp = np.zeros((self.grid.qx+1, self.grid.qy, self.nvar),
                            dtype=self.dtype)
        elif self.idir == 2:
            _tmp = np.zeros((self.grid.qx, self.grid.qy+1, self.nvar),
                            dtype=self.dtype)

        self.data = ArrayIndexerFC(_tmp, idir=self.idir, grid=self.grid)

        self.initialized = 1

    def __str__(self):
        """ print out some basic information about the FaceCenterData2d
            object """

        if self.initialized == 0:
            my_str = "FaceCenterData2d object not yet initialized"
            return my_str

        my_str = f"fc data: idir = {self.idir}, nx = {self.grid.nx}, ny = {self.grid.ny}, ng = {self.grid.ng}\n"
        my_str += f"         nvars = {self.nvar}\n"
        my_str += "         variables:\n"

        for n in range(self.nvar):
            name = self.names[n]
            my_str += f"{name:>16s}: min: {self.min(name):15.10f}    max: {self.max(name):15.10f}\n"
            my_str += f"{' ':>16s}  BCs: -x: {self.BCs[name].xlb:12s} +x: {self.BCs[name].xrb:12s}"
            my_str += f" -y: {self.BCs[name].ylb:12s} +y: {self.BCs[name].yrb:12s}\n"

        return my_str

    def get_var_by_index(self, n):
        """
        Return a data array for the variable with index n in the
        data array.  Any changes made to this are automatically
        reflected in the CellCenterData2d object.

        Parameters
        ----------
        n : int
            The index of the variable to access

        Returns
        -------
        out : ndarray
            The array of data corresponding to the index

        """
        return ArrayIndexerFC(d=self.data[:, :, n], idir=self.idir, grid=self.grid)

    def get_vars(self):
        """
        Return the entire data array.  Any changes made to this
        are automatically reflected in the CellCenterData2d object.

        Returns
        -------
        out : ndarray
            The array of data

        """
        return ArrayIndexerFC(d=self.data, idir=self.idir, grid=self.grid)

    def fill_BC(self, name):
        """
        Fill the boundary conditions.  This operates on a single state
        variable at a time, to allow for maximum flexibility.

        We do periodic, reflect-even, reflect-odd, and outflow

        Each variable name has a corresponding BC stored in the
        CellCenterData2d object -- we refer to this to figure out the
        action to take at each boundary.

        Parameters
        ----------
        name : str
            The name of the variable for which to fill the BCs.

        """

        n = self.names.index(name)
        self.data.fill_ghost(n=n, bc=self.BCs[name])

        if self.BCs[name].xlb in bnd.ext_bcs or \
           self.BCs[name].xrb in bnd.ext_bcs or \
           self.BCs[name].ylb in bnd.ext_bcs or \
           self.BCs[name].yrb in bnd.ext_bcs:
            raise NotImplementedError("custom boundary conditions not supported for FaceCenterData2d")

    def restrict(self, varname, N=2):
        raise NotImplementedError("restriction not implemented for FaceCenterData2d")

    def prolong(self, varname):
        raise NotImplementedError("prolongation not implemented for FaceCenterData2d")

    def write_data(self, f):
        """
        write the data out to an hdf5 file -- here, f is an h5py
        File pbject

        """

        # data
        gstate = f.create_group("face-centered-state")

        for n in range(self.nvar):
            gvar = gstate.create_group(self.names[n])
            gvar.create_dataset("data",
                                data=self.get_var_by_index(n).v())
            gvar.attrs["xlb"] = self.BCs[self.names[n]].xlb
            gvar.attrs["xrb"] = self.BCs[self.names[n]].xrb
            gvar.attrs["ylb"] = self.BCs[self.names[n]].ylb
            gvar.attrs["yrb"] = self.BCs[self.names[n]].yrb


def cell_center_data_clone(old):
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

    if not isinstance(old, CellCenterData2d):
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

    return new


def do_demo():
    """ show examples of the patch methods / classes """

    import pyro.util.io_pyro as io

    # illustrate basic mesh operations

    myg = Grid2d(8, 16, xmax=1.0, ymax=2.0)

    mydata = CellCenterData2d(myg)

    bc = bnd.BC()

    mydata.register_var("a", bc)
    mydata.create()

    a = mydata.get_var("a")
    a[:, :] = np.exp(-(myg.x2d - 0.5)**2 - (myg.y2d - 1.0)**2)

    print(mydata)

    # output
    print("writing\n")
    mydata.write("mesh_test")

    print("reading\n")
    myd2 = io.read("mesh_test")
    print(myd2)

    mydata.pretty_print("a")


if __name__ == "__main__":
    do_demo()
