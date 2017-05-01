"""
The patch module defines the classes necessary to describe finite-volume
data and the grid that it lives on.

Typical usage:

  -- create the grid

     grid = Grid2d(nx, ny)


  -- create the data that lives on that grid

     data = CellCenterData2d(grid)

     bc = BC(xlb="reflect", xrb="reflect",
             ylb="outflow", yrb="outflow")
     data.register_var("density", bc)
     ...

     data.create()


  -- initialize some data

     dens = data.get_var("density")
     dens[:,:] = ...


  -- fill the ghost cells

     data.fill_BC("density")

"""
from __future__ import print_function

import numpy as np
import pickle

import h5py

from util import msg

import mesh.boundary as bnd
import mesh.array_indexer as ai


class Grid2d(object):
    """
    the 2-d grid class.  The grid object will contain the coordinate
    information (at various centerings).

    A basic (1-d) representation of the layout is:

    |     |      |     X     |     |      |     |     X     |      |     |
    +--*--+- // -+--*--X--*--+--*--+- // -+--*--+--*--X--*--+- // -+--*--+
       0          ng-1    ng   ng+1         ... ng+nx-1 ng+nx      2ng+nx-1

                         ilo                      ihi

    |<- ng guardcells->|<---- nx interior zones ----->|<- ng guardcells->|

    The '*' marks the data locations.
    """

    def __init__(self, nx, ny, ng=1, \
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
        return ai.ArrayIndexer(d=_tmp, grid=self)


    def norm(self, d):
        """
        find the norm of the quantity d defined on the same grid, in the
        domain's valid region
        """
        return np.sqrt(self.dx*self.dy*
                       np.sum((d[self.ilo:self.ihi+1,self.jlo:self.jhi+1]**2).flat))


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
        return "2-d grid: nx = {}, ny = {}, ng = {}".format(
            self.nx, self.ny, self.ng)


    def __eq__(self, other):
        """ are two grids equivalent? """
        result = (self.nx == other.nx and self.ny == other.ny and
                  self.ng == other.ng and
                  self.xmin == other.xmin and self.xmax == other.xmax and
                  self.ymin == other.ymin and self.ymax == other.ymax)

        return result


class CellCenterData2d(object):
    """
    A class to define cell-centered data that lives on a grid.  A
    CellCenterData2d object is built in a multi-step process before
    it can be used.

    -- Create the object.  We pass in a grid object to describe where
       the data lives:

       my_data = patch.CellCenterData2d(myGrid)

    -- Register any variables that we expect to live on this patch.
       Here BC describes the boundary conditions for that variable.

       my_data.register_var('density', BC)
       my_data.register_var('x-momentum', BC)
       ...

    -- Register any auxillary data -- these are any parameters that are
       needed to interpret the data outside of the simulation (for
       example, the gamma for the equation of state).

       my_data.set_aux(keyword, value)

    -- Finish the initialization of the patch

       my_data.create()

    This last step actually allocates the storage for the state
    variables.  Once this is done, the patch is considered to be
   locked.  New variables cannot be added.
    """

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
        runtime_parameters : RuntimeParameters object, optional
            The runtime parameters that go along with this data

        """

        self.grid = grid

        self.dtype = dtype
        self.data = None

        self.names = []
        self.vars = self.names # backwards compatibility hack
        self.nvar = 0

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


    def create(self):
        """
        Called after all the variables are registered and allocates
        the storage for the state data.
        """

        if self.initialized == 1:
            msg.fail("ERROR: grid already initialized")

        self.data = np.zeros((self.grid.qx, self.grid.qy, self.nvar),
                                dtype=self.dtype)
        self.initialized = 1


    def __str__(self):
        """ print out some basic information about the CellCenterData2d
            object """

        if self.initialized == 0:
            my_str = "CellCenterData2d object not yet initialized"
            return my_str

        my_str = "cc data: nx = {}, ny = {}, ng = {}\n".format(
            self.grid.nx, self.grid.ny, self.grid.ng)
        my_str += "         nvars = {}\n".format(self.nvar)
        my_str += "         variables:\n"

        ilo = self.grid.ilo
        ihi = self.grid.ihi
        jlo = self.grid.jlo
        jhi = self.grid.jhi

        for n in range(self.nvar):
            my_str += "%16s: min: %15.10f    max: %15.10f\n" % \
                (self.names[n], self.min(self.names[n]), self.max(self.names[n]))
            my_str += "%16s  BCs: -x: %-12s +x: %-12s -y: %-12s +y: %-12s\n" %\
                (" " , self.BCs[self.names[n]].xlb,
                       self.BCs[self.names[n]].xrb,
                       self.BCs[self.names[n]].ylb,
                       self.BCs[self.names[n]].yrb)

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
        try:
            n = self.names.index(name)
        except:
            for f in self.derives:
                var = f(self, name)
                if len(var) > 0:
                    return var
            raise KeyError("name {} is not valid".format(name))
        else:
            return ai.ArrayIndexer(d=self.data[:,:,n], grid=self.grid)


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
        return ai.ArrayIndexer(d=self.data[:,:,n], grid=self.grid)


    def get_vars(self):
        """
        Return the entire data array.  Any changes made to this
        are automatically reflected in the CellCenterData2d object.

        Returns
        -------
        out : ndarray
            The array of data

        """
        return ai.ArrayIndexer(d=self.data, grid=self.grid)


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
        if keyword in self.aux.keys():
            return self.aux[keyword]
        else:
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
        self.data[:,:,n] = 0.0


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

        # there is only a single grid, so every boundary is on
        # a physical boundary (except if we are periodic)

        # Note: we piggy-back on outflow and reflect-odd for
        # Neumann and Dirichlet homogeneous BCs respectively, but
        # this only works for a single ghost cell

        n = self.names.index(name)

        # -x boundary
        if self.BCs[name].xlb in ["outflow", "neumann"]:

            if self.BCs[name].xl_value is None:
                for i in range(self.grid.ilo):
                    self.data[i,:,n] = self.data[self.grid.ilo,:,n]
            else:
                self.data[self.grid.ilo-1,:,n] = \
                    self.data[self.grid.ilo,:,n] - self.grid.dx*self.BCs[name].xl_value[:]

        elif self.BCs[name].xlb == "reflect-even":

            for i in range(self.grid.ilo):
                self.data[i,:,n] = self.data[2*self.grid.ng-i-1,:,n]

        elif self.BCs[name].xlb in ["reflect-odd", "dirichlet"]:

            if self.BCs[name].xl_value is None:
                for i in range(self.grid.ilo):
                    self.data[i,:,n] = -self.data[2*self.grid.ng-i-1,:,n]
            else:
                self.data[self.grid.ilo-1,:,n] = \
                    2*self.BCs[name].xl_value[:] - self.data[self.grid.ilo,:,n]

        elif self.BCs[name].xlb == "periodic":

            for i in range(self.grid.ilo):
                self.data[i,:,n] = self.data[self.grid.ihi-self.grid.ng+i+1,:,n]


        # +x boundary
        if self.BCs[name].xrb in ["outflow", "neumann"]:

            if self.BCs[name].xr_value is None:
                for i in range(self.grid.ihi+1, self.grid.nx+2*self.grid.ng):
                    self.data[i,:,n] = self.data[self.grid.ihi,:,n]
            else:
                self.data[self.grid.ihi+1,:,n] = \
                    self.data[self.grid.ihi,:,n] + self.grid.dx*self.BCs[name].xr_value[:]

        elif self.BCs[name].xrb == "reflect-even":

            for i in range(self.grid.ng):
                i_bnd = self.grid.ihi+1+i
                i_src = self.grid.ihi-i

                self.data[i_bnd,:,n] = self.data[i_src,:,n]

        elif self.BCs[name].xrb in ["reflect-odd", "dirichlet"]:

            if self.BCs[name].xr_value is None:
                for i in range(self.grid.ng):
                    i_bnd = self.grid.ihi+1+i
                    i_src = self.grid.ihi-i

                    self.data[i_bnd,:,n] = -self.data[i_src,:,n]
            else:
                self.data[self.grid.ihi+1,:,n] = \
                    2*self.BCs[name].xr_value[:] - self.data[self.grid.ihi,:,n]

        elif self.BCs[name].xrb == "periodic":

            for i in range(self.grid.ihi+1, 2*self.grid.ng + self.grid.nx):
                self.data[i,:,n] = self.data[i-self.grid.ihi-1+self.grid.ng,:,n]


        # -y boundary
        if self.BCs[name].ylb in ["outflow", "neumann"]:

            if self.BCs[name].yl_value is None:
                for j in range(self.grid.jlo):
                    self.data[:,j,n] = self.data[:,self.grid.jlo,n]
            else:
                self.data[:,self.grid.jlo-1,n] = \
                    self.data[:,self.grid.jlo,n] - self.grid.dy*self.BCs[name].yl_value[:]

        elif self.BCs[name].ylb == "reflect-even":

            for j in range(self.grid.jlo):
                self.data[:,j,n] = self.data[:,2*self.grid.ng-j-1,n]

        elif self.BCs[name].ylb in ["reflect-odd", "dirichlet"]:

            if self.BCs[name].yl_value is None:
                for j in range(self.grid.jlo):
                    self.data[:,j,n] = -self.data[:,2*self.grid.ng-j-1,n]
            else:
                self.data[:,self.grid.jlo-1,n] = \
                    2*self.BCs[name].yl_value[:] - self.data[:,self.grid.jlo,n]

        elif self.BCs[name].ylb == "periodic":

            for j in range(self.grid.jlo):
                self.data[:,j,n] = self.data[:,self.grid.jhi-self.grid.ng+j+1,n]

        else:
            if self.BCs[name].ylb in bnd.ext_bcs.keys():

                bnd.ext_bcs[self.BCs[name].ylb](self.BCs[name].ylb, "ylb", name, self)


        # +y boundary
        if self.BCs[name].yrb in ["outflow", "neumann"]:

            if self.BCs[name].yr_value is None:
                for j in range(self.grid.jhi+1, self.grid.ny+2*self.grid.ng):
                    self.data[:,j,n] = self.data[:,self.grid.jhi,n]
            else:
                self.data[:,self.grid.jhi+1,n] = \
                    self.data[:,self.grid.jhi,n] + self.grid.dy*self.BCs[name].yr_value[:]

        elif self.BCs[name].yrb == "reflect-even":

            for j in range(self.grid.ng):
                j_bnd = self.grid.jhi+1+j
                j_src = self.grid.jhi-j

                self.data[:,j_bnd,n] = self.data[:,j_src,n]

        elif self.BCs[name].yrb in ["reflect-odd", "dirichlet"]:

            if self.BCs[name].yr_value is None:
                for j in range(self.grid.ng):
                    j_bnd = self.grid.jhi+1+j
                    j_src = self.grid.jhi-j

                    self.data[:,j_bnd,n] = -self.data[:,j_src,n]
            else:
                self.data[:,self.grid.jhi+1,n] = \
                    2*self.BCs[name].yr_value[:] - self.data[:,self.grid.jhi,n]

        elif self.BCs[name].yrb == "periodic":

            for j in range(self.grid.jhi+1, 2*self.grid.ng + self.grid.ny):
                self.data[:,j,n] = self.data[:,j-self.grid.jhi-1+self.grid.ng,n]

        else:
            if self.BCs[name].yrb in bnd.ext_bcs.keys():

                bnd.ext_bcs[self.BCs[name].yrb](self.BCs[name].yrb, "yrb", name, self)


    def min(self, name, ng=0):
        """
        return the minimum of the variable name in the domain's valid region
        """
        n = self.names.index(name)
        g = self.grid
        return np.min(self.data[g.ilo-ng:g.ihi+1+ng,g.jlo-ng:g.jhi+1+ng,n])


    def max(self, name, ng=0):
        """
        return the maximum of the variable name in the domain's valid region
        """
        n = self.names.index(name)
        g = self.grid
        return np.max(self.data[g.ilo-ng:g.ihi+1+ng,g.jlo-ng:g.jhi+1+ng,n])


    def restrict(self, varname):
        """
        Restrict the variable varname to a coarser grid (factor of 2
        coarser) and return an array with the resulting data (and same
        number of ghostcells)
        """

        fine_grid = self.grid
        fdata = self.get_var(varname)

        # allocate an array for the coarsely gridded data
        coarse_grid = fine_grid.coarse_like(2)
        cdata = coarse_grid.scratch_array()

        # fill the coarse array with the restricted data -- just
        # average the 4 fine cells into the corresponding coarse cell
        # that encompasses them.
        cdata.v()[:,:] = \
            0.25*(fdata.v(s=2) + fdata.ip(1, s=2) +
                  fdata.jp(1, s=2) + fdata.ip_jp(1, 1, s=2))

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
        independently monotonic:

                  (x)         (y)
        f(x,y) = m    x/dx + m    y/dy + <f>

        where the m's are the limited differences in each direction.
        When averaged over the parent cell, this reproduces <f>.

        Each zone's reconstrution will be averaged over 4 children.

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
        m_x.v()[:,:] = 0.5*(cdata.ip(1) - cdata.ip(-1))

        m_y = coarse_grid.scratch_array()
        m_y.v()[:,:] = 0.5*(cdata.jp(1) - cdata.jp(-1))

        # fill the children
        fdata.v(s=2)[:,:] = cdata.v() - 0.25*m_x.v() - 0.25*m_y.v()     # 1 child
        fdata.ip(1, s=2)[:,:] = cdata.v() + 0.25*m_x.v() - 0.25*m_y.v() # 2
        fdata.jp(1, s=2)[:,:] = cdata.v() - 0.25*m_x.v() + 0.25*m_y.v() # 3
        fdata.ip_jp(1, 1, s=2)[:,:] = cdata.v() + 0.25*m_x.v() + 0.25*m_y.v() # 4

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

        a = self.get_var(var)
        a.pretty_print(fmt=fmt)



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

    new = CellCenterData2d(old.grid, dtype=old.dtype)

    for n in range(old.nvar):
        new.register_var(old.names[n], old.BCs[old.names[n]])

    new.create()

    new.aux = old.aux.copy()
    new.data = old.data.copy()
    new.derives = old.derives.copy()

    return new


def do_demo():

    import util.io as io

    # illustrate basic mesh operations

    myg = Grid2d(8, 16, xmax=1.0, ymax=2.0)

    mydata = CellCenterData2d(myg)

    bc = bnd.BC()

    mydata.register_var("a", bc)
    mydata.create()


    a = mydata.get_var("a")
    a[:,:] = np.exp(-(myg.x2d - 0.5)**2 - (myg.y2d - 1.0)**2)

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
