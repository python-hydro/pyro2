"""
The patch module defines the classes necessary to describe finite-volume
data and the grid that it lives on.

Typical usage:

  -- create the grid

     grid = Grid2d(nx, ny)


  -- create the data that lives on that grid

     data = CellCenterData2d(grid)

     bc = BCObject(xlb="reflect", xrb="reflect",
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

from util import msg

valid = ["outflow", "periodic",
         "reflect", "reflect-even", "reflect-odd",
         "dirichlet", "neumann"]

extBCs = {}

def define_bc(type, function):
    """
    use this to extend the types of boundary conditions supported
    on a solver-by-solver basis.  Here we pass in the reference to
    a function that can be called with the data that needs to be
    filled.
    """

    valid.append(type)
    extBCs[type] = function


class BCObject:
    """Boundary condition container -- hold the BCs on each boundary
    for a single variable.  

    For Neumann and Dirichlet BCs, a function callback can be stored
    for inhomogeous BCs.  This function should provide the value on
    the physical boundary (not cell center).  Note: this only
    ensures that the first ghost cells is consistent with the BC
    value.

    """

    def __init__ (self,
                  xlb="outflow", xrb="outflow",
                  ylb="outflow", yrb="outflow",
                  xl_func=None, xr_func=None,
                  yl_func=None, yr_func=None,
                  odd_reflect_dir=""):
        """
        Create the BCObject.

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

        """

        # note: "reflect" is ambiguous and will be converted into
        # either reflect-even (the default) or reflect-odd if
        # odd_reflect_dir specifies the corresponding direction ("x",
        # "y")

        # -x boundary
        if xlb in valid:
            self.xlb = xlb
            if self.xlb == "reflect":
                if odd_reflect_dir == "x":
                    self.xlb = "reflect-odd"
                else:
                    self.xlb = "reflect-even"

        else:
            msg.fail("ERROR: xlb = %s invalid BC" % (xlb))

        # +x boundary
        if xrb in valid:
            self.xrb = xrb
            if self.xrb == "reflect":
                if odd_reflect_dir == "x":
                    self.xrb = "reflect-odd"
                else:
                    self.xrb = "reflect-even"

        else:
            msg.fail("ERROR: xrb = %s invalid BC" % (xrb))

        # -y boundary
        if ylb in valid:
            self.ylb = ylb
            if self.ylb == "reflect":
                if odd_reflect_dir == "y":
                    self.ylb = "reflect-odd"
                else:
                    self.ylb = "reflect-even"

        else:
            msg.fail("ERROR: ylb = %s invalid BC" % (ylb))

        # +y boundary
        if yrb in valid:
            self.yrb = yrb
            if self.yrb == "reflect":
                if odd_reflect_dir == "y":
                    self.yrb = "reflect-odd"
                else:
                    self.yrb = "reflect-even"

        else:
            msg.fail("ERROR: yrb = %s invalid BC" % (yrb))


        # periodic checks
        if ((xlb == "periodic" and not xrb == "periodic") or
            (xrb == "periodic" and not xlb == "periodic")):
            msg.fail("ERROR: both xlb and xrb must be periodic")

        if ((ylb == "periodic" and not yrb == "periodic") or
            (yrb == "periodic" and not ylb == "periodic")):
            msg.fail("ERROR: both ylb and yrb must be periodic")


        # inhomogeneous functions for Dirichlet or Neumann
        self.xl_func = xl_func
        self.xr_func = xr_func
        self.yl_func = yl_func
        self.yr_func = yr_func        
        
            
    def __str__(self):
        """ print out some basic information about the BC object """

        string = "BCs: -x: %s  +x: %s  -y: %s  +y: %s" % \
            (self.xlb, self.xrb, self.ylb, self.yrb)

        return string


class Grid2d:
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

    def __init__ (self, nx, ny, ng=1, \
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
        self.nx = nx
        self.ny = ny
        self.ng = ng

        self.qx = 2*ng+nx
        self.qy = 2*ng+ny

        # domain extrema
        self.xmin = xmin
        self.xmax = xmax

        self.ymin = ymin
        self.ymax = ymax

        # compute the indices of the block interior (excluding guardcells)
        self.ilo = ng
        self.ihi = ng+nx-1

        self.jlo = ng
        self.jhi = ng+ny-1

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
            return np.zeros((self.qx, self.qy), dtype=np.float64)
        else:
            return np.zeros((self.qx, self.qy, nvar), dtype=np.float64)


    def coarse_like(self, N):
        """
        return a new grid object coarsened by a factor n, but with
        all the other properties the same
        """
        return Grid2d(self.nx/N, self.ny/N, ng=self.ng,
                      xmin=self.xmin, xmax=self.xmax,
                      ymin=self.ymin, ymax=self.ymax)


    def __str__(self):
        """ print out some basic information about the grid object """
        return "2-d grid: nx = {}, ny = {}, ng = {}".format(self.nx, self.ny,self.ng)


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

    -- Create the object.  We pass in a grid object to describe where
       the data lives:

       my_data = patch.CellCenterData2d(myGrid)

    -- Register any variables that we expect to live on this patch.
       Here BCObject describes the boundary conditions for that variable.

       my_data.register_var('density', BCObject)
       my_data.register_var('x-momentum', BCObject)
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

    def __init__ (self, grid, dtype=np.float64):

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

        self.vars = []
        self.nvar = 0

        self.aux = {}

        self.BCs = {}

        # time
        self.t = -1.0

        self.initialized = 0


    def register_var(self, name, bc_object):
        """
        Register a variable with CellCenterData2d object.

        Parameters
        ----------
        name : str
            The variable name
        bc_object : BCObject object
            The boundary conditions that describe the actions to take
            for this variable at the physical domain boundaries.
        """

        if self.initialized == 1:
            msg.fail("ERROR: grid already initialized")

        self.vars.append(name)
        self.nvar += 1

        self.BCs[name] = bc_object


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


    def create(self):
        """
        Called after all the variables are registered and allocates
        the storage for the state data.
        """

        if self.initialized == 1:
            msg.fail("ERROR: grid already initialized")

        self.data = np.zeros((self.nvar, self.grid.qx, self.grid.qy),
                                dtype=self.dtype)
        self.initialized = 1


    def __str__(self):
        """ print out some basic information about the CellCenterData2d
            object """

        if self.initialized == 0:
            myStr = "CellCenterData2d object not yet initialized"
            return myStr

        myStr = "cc data: nx = " + repr(self.grid.nx) + \
                       ", ny = " + repr(self.grid.ny) + \
                       ", ng = " + repr(self.grid.ng) + "\n" + \
                 "   nvars = " + repr(self.nvar) + "\n" + \
                 "   variables: \n"

        ilo = self.grid.ilo
        ihi = self.grid.ihi
        jlo = self.grid.jlo
        jhi = self.grid.jhi

        n = 0
        while n < self.nvar:
            myStr += "%16s: min: %15.10f    max: %15.10f\n" % \
                (self.vars[n],
                 np.min(self.data[n,ilo:ihi+1,jlo:jhi+1]),
                 np.max(self.data[n,ilo:ihi+1,jlo:jhi+1]) )
            myStr += "%16s  BCs: -x: %-12s +x: %-12s -y: %-12s +y: %-12s\n" %\
                (" " , self.BCs[self.vars[n]].xlb,
                       self.BCs[self.vars[n]].xrb,
                       self.BCs[self.vars[n]].ylb,
                       self.BCs[self.vars[n]].yrb)
            n += 1

        return myStr


    def get_var(self, name):
        """
        Return a data array for the variable described by name.
        Any changes made to this are automatically reflected in the
        CellCenterData2d object.

        Parameters
        ----------
        name : str
            The name of the variable to access

        Returns
        -------
        out : ndarray
            The array of data corresponding to the variable name

        """
        n = self.vars.index(name)
        return self.data[n,:,:]


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
        return self.data[n,:,:]


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
        n = self.vars.index(name)
        self.data[n,:,:] = 0.0


    def fill_BC_all(self):
        """
        Fill boundary conditions on all variables.
        """
        for name in self.vars:
            self.fill_BC(name)


    def fill_BC(self, name):
        """
        Fill the boundary conditions.  This operates on a single state
        variable at a time, to allow for maximum flexibility.

        We do periodic, reflect-even, reflect-odd, and outflow

        Each variable name has a corresponding BCObject stored in the
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

        n = self.vars.index(name)

        # -x boundary
        if self.BCs[name].xlb in ["outflow", "neumann"]:

            if self.BCs[name].xl_func == None:            
                for i in range(self.grid.ilo):
                    self.data[n,i,:] = self.data[n,self.grid.ilo,:]
            else:
                self.data[n,self.grid.ilo-1,:] = \
                    self.data[n,self.grid.ilo,:] - self.grid.dx*self.BCs[name].xl_func(self.grid.y)
                
        elif self.BCs[name].xlb == "reflect-even":

            for i in range(self.grid.ilo):
                self.data[n,i,:] = self.data[n,2*self.grid.ng-i-1,:]

        elif self.BCs[name].xlb in ["reflect-odd", "dirichlet"]:

            if self.BCs[name].xl_func == None:
                for i in range(self.grid.ilo):
                    self.data[n,i,:] = -self.data[n,2*self.grid.ng-i-1,:]
            else:
                self.data[n,self.grid.ilo-1,:] = \
                    2*self.BCs[name].xl_func(self.grid.y) - self.data[n,self.grid.ilo,:]

        elif self.BCs[name].xlb == "periodic":

            for i in range(self.grid.ilo):
                self.data[n,i,:] = self.data[n,self.grid.ihi-self.grid.ng+i+1,:]


        # +x boundary
        if self.BCs[name].xrb in ["outflow", "neumann"]:

            if self.BCs[name].xr_func == None:
                for i in range(self.grid.ihi+1, self.grid.nx+2*self.grid.ng):
                    self.data[n,i,:] = self.data[n,self.grid.ihi,:]
            else:
                self.data[n,self.grid.ihi+1,:] = \
                    self.data[n,self.grid.ihi,:] + self.grid.dx*self.BCs[name].xr_func(self.grid.y)

        elif self.BCs[name].xrb == "reflect-even":

            for i in range(self.grid.ng):
                i_bnd = self.grid.ihi+1+i
                i_src = self.grid.ihi-i

                self.data[n,i_bnd,:] = self.data[n,i_src,:]

        elif self.BCs[name].xrb in ["reflect-odd", "dirichlet"]:

            if self.BCs[name].xr_func == None:
                for i in range(self.grid.ng):
                    i_bnd = self.grid.ihi+1+i
                    i_src = self.grid.ihi-i

                    self.data[n,i_bnd,:] = -self.data[n,i_src,:]
            else:
                self.data[n,self.grid.ihi+1,:] = \
                    2*self.BCs[name].xr_func(self.grid.y) - self.data[n,self.grid.ihi,:]

        elif self.BCs[name].xrb == "periodic":

            for i in range(self.grid.ihi+1, 2*self.grid.ng + self.grid.nx):
                self.data[n,i,:] = self.data[n,i-self.grid.ihi-1+self.grid.ng,:]


        # -y boundary
        if self.BCs[name].ylb in ["outflow", "neumann"]:

            if self.BCs[name].yl_func == None:
                for j in range(self.grid.jlo):
                    self.data[n,:,j] = self.data[n,:,self.grid.jlo]
            else:
                self.data[n,:,self.grid.jlo-1] = \
                    self.data[n,:,self.grid.jlo] - self.grid.dx*self.BCs[name].yl_func(self.grid.x)                    

        elif self.BCs[name].ylb == "reflect-even":

            for j in range(self.grid.jlo):
                self.data[n,:,j] = self.data[n,:,2*self.grid.ng-j-1]

        elif self.BCs[name].ylb in ["reflect-odd", "dirichlet"]:

            if self.BCs[name].yl_func == None:
                for j in range(self.grid.jlo):
                    self.data[n,:,j] = -self.data[n,:,2*self.grid.ng-j-1]
            else:
                self.data[n,:,self.grid.jlo-1] = \
                    2*self.BCs[name].yl_func(self.grid.x) - self.data[n,:,self.grid.jlo]
                
        elif self.BCs[name].ylb == "periodic":

            for j in range(self.grid.jlo):
                self.data[n,:,j] = self.data[n,:,self.grid.jhi-self.grid.ng+j+1]

        else:
            if self.BCs[name].ylb in extBCs.keys():

                extBCs[self.BCs[name].ylb](self.BCs[name].ylb, "ylb", name, self)


        # +y boundary
        if self.BCs[name].yrb in ["outflow", "neumann"]:

            if self.BCs[name].yr_func == None:
                for j in range(self.grid.jhi+1, self.grid.ny+2*self.grid.ng):
                    self.data[n,:,j] = self.data[n,:,self.grid.jhi]
            else:
                self.data[n,:,self.grid.jhi+1] = \
                    self.data[n,:,self.grid.jhi] + self.grid.dx*self.BCs[name].yr_func(self.grid.x)                

        elif self.BCs[name].yrb == "reflect-even":

            for j in range(self.grid.ng):
                j_bnd = self.grid.jhi+1+j
                j_src = self.grid.jhi-j

                self.data[n,:,j_bnd] = self.data[n,:,j_src]

        elif self.BCs[name].yrb in ["reflect-odd", "dirichlet"]:

            if self.BCs[name].yr_func == None:
                for j in range(self.grid.ng):
                    j_bnd = self.grid.jhi+1+j
                    j_src = self.grid.jhi-j

                    self.data[n,:,j_bnd] = -self.data[n,:,j_src]
            else:
                self.data[n,:,self.grid.jhi+1] = \
                    2*self.BCs[name].yr_func(self.grid.x) - self.data[n,:,self.grid.jhi]
                
        elif self.BCs[name].yrb == "periodic":

            for j in range(self.grid.jhi+1, 2*self.grid.ng + self.grid.ny):
                self.data[n,:,j] = self.data[n,:,j-self.grid.jhi-1+self.grid.ng]

        else:
            if self.BCs[name].yrb in extBCs.keys():

                extBCs[self.BCs[name].yrb](self.BCs[name].yrb, "yrb", name, self)


    def min(self, name, ng=0):
        """
        return the minimum of the variable name in the domain's valid region
        """
        n = self.vars.index(name)
        g = self.grid
        return np.min(self.data[n,g.ilo-ng:g.ihi+1+ng,g.jlo-ng:g.jhi+1+ng])

    
    def max(self, name, ng=0):
        """
        return the maximum of the variable name in the domain's valid region
        """
        n = self.vars.index(name)
        g = self.grid
        return np.max(self.data[n,g.ilo-ng:g.ihi+1+ng,g.jlo-ng:g.jhi+1+ng])
    
    
    def restrict(self, varname):
        """
        Restrict the variable varname to a coarser grid (factor of 2
        coarser) and return an array with the resulting data (and same
        number of ghostcells)
        """

        fG = self.grid
        fData = self.get_var(varname)

        # allocate an array for the coarsely gridded data
        ng_c = fG.ng
        nx_c = fG.nx/2
        ny_c = fG.ny/2

        cData = np.zeros((2*ng_c+nx_c, 2*ng_c+ny_c), dtype=self.dtype)

        ilo_c = ng_c
        ihi_c = ng_c+nx_c-1

        jlo_c = ng_c
        jhi_c = ng_c+ny_c-1

        # fill the coarse array with the restricted data -- just
        # average the 4 fine cells into the corresponding coarse cell
        # that encompasses them.

        # This is done by shifting our view into the fData array and
        # using a stride of 2 in the indexing.
        cData[ilo_c:ihi_c+1,jlo_c:jhi_c+1] = \
            0.25*(fData[fG.ilo  :fG.ihi+1:2,fG.jlo  :fG.jhi+1:2] +
                  fData[fG.ilo+1:fG.ihi+1:2,fG.jlo  :fG.jhi+1:2] +
                  fData[fG.ilo  :fG.ihi+1:2,fG.jlo+1:fG.jhi+1:2] +
                  fData[fG.ilo+1:fG.ihi+1:2,fG.jlo+1:fG.jhi+1:2])

        return cData


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

        cG = self.grid
        cData = self.get_var(varname)

        # allocate an array for the coarsely gridded data
        ng_f = cG.ng
        nx_f = cG.nx*2
        ny_f = cG.ny*2

        fData = np.zeros((2*ng_f+nx_f, 2*ng_f+ny_f), dtype=self.dtype)

        ilo_f = ng_f
        ihi_f = ng_f+nx_f-1

        jlo_f = ng_f
        jhi_f = ng_f+ny_f-1

        # slopes for the coarse data
        m_x = cG.scratch_array()
        m_x[cG.ilo:cG.ihi+1,cG.jlo:cG.jhi+1] = \
            0.5*(cData[cG.ilo+1:cG.ihi+2,cG.jlo:cG.jhi+1] -
                 cData[cG.ilo-1:cG.ihi  ,cG.jlo:cG.jhi+1])

        m_y = cG.scratch_array()
        m_y[cG.ilo:cG.ihi+1,cG.jlo:cG.jhi+1] = \
            0.5*(cData[cG.ilo:cG.ihi+1,cG.jlo+1:cG.jhi+2] -
                 cData[cG.ilo:cG.ihi+1,cG.jlo-1:cG.jhi  ])



        # fill the '1' children
        fData[ilo_f:ihi_f+1:2,jlo_f:jhi_f+1:2] = \
            cData[cG.ilo:cG.ihi+1,cG.jlo:cG.jhi+1] \
            - 0.25*m_x[cG.ilo:cG.ihi+1,cG.jlo:cG.jhi+1] \
            - 0.25*m_y[cG.ilo:cG.ihi+1,cG.jlo:cG.jhi+1]


        # fill the '2' children
        fData[ilo_f+1:ihi_f+1:2,jlo_f:jhi_f+1:2] = \
            cData[cG.ilo:cG.ihi+1,cG.jlo:cG.jhi+1] \
            + 0.25*m_x[cG.ilo:cG.ihi+1,cG.jlo:cG.jhi+1] \
            - 0.25*m_y[cG.ilo:cG.ihi+1,cG.jlo:cG.jhi+1]


        # fill the '3' children
        fData[ilo_f:ihi_f+1:2,jlo_f+1:jhi_f+1:2] = \
            cData[cG.ilo:cG.ihi+1,cG.jlo:cG.jhi+1] \
            - 0.25*m_x[cG.ilo:cG.ihi+1,cG.jlo:cG.jhi+1] \
            + 0.25*m_y[cG.ilo:cG.ihi+1,cG.jlo:cG.jhi+1]


        # fill the '4' children
        fData[ilo_f+1:ihi_f+1:2,jlo_f+1:jhi_f+1:2] = \
            cData[cG.ilo:cG.ihi+1,cG.jlo:cG.jhi+1] \
            + 0.25*m_x[cG.ilo:cG.ihi+1,cG.jlo:cG.jhi+1] \
            + 0.25*m_y[cG.ilo:cG.ihi+1,cG.jlo:cG.jhi+1]

        return fData


    def write(self, filename):
        """
        write out the CellCenterData2d object to disk, stored in the
        file filename.  We use a python binary format (via pickle).
        This stores a representation of the entire object.
        """
        pF = open(filename + ".pyro", "wb")
        pickle.dump(self, pF, pickle.HIGHEST_PROTOCOL)
        pF.close()


    def pretty_print(self, varname):
        """
        Print out a small dataset to the screen with the ghost cells
        a different color, to make things stand out
        """

        a = self.get_var(varname)

        if self.dtype == np.int:
            fmt = "%4d"
        elif self.dtype == np.float64:
            fmt = "%10.5g"
        else:
            msg.fail("ERROR: dtype not supported")

        # print j descending, so it looks like a grid (y increasing
        # with height)
        j = self.grid.qy-1
        while j >= 0:
            i = 0
            while i < self.grid.qx:

                if (j < self.grid.jlo or j > self.grid.jhi or
                    i < self.grid.ilo or i > self.grid.ihi):
                    gc = 1
                else:
                    gc = 0

                if gc:
                    print("\033[31m" + fmt % (a[i,j]) + "\033[0m", end="")
                else:
                    print (fmt % (a[i,j]), end="")

                i += 1

            print(" ")
            j -= 1

        leg = """
         ^ y
         |
         +---> x
        """
        print(leg)


# backwards compatibility
ccData2d = CellCenterData2d
grid2d = Grid2d
bcObject = BCObject

def read(filename):
    """
    Read a CellCenterData object from a file and return it and the grid
    info and data.
    """

    # if we come in with .pyro, we don't need to add it again
    if filename.find(".pyro") < 0:
        filename += ".pyro"

    pF = open(filename, "rb")
    data = pickle.load(pF)
    pF.close()

    return data.grid, data


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

    n = 0
    while n < old.nvar:
        new.register_var(old.vars[n], old.BCs[old.vars[n]])
        n += 1

    new.create()

    new.aux = old.aux.copy()
    new.data = old.data.copy()

    return new


if __name__== "__main__":

    # illustrate basic mesh operations

    myg = Grid2d(16,32, xmax=1.0, ymax=2.0)

    mydata = CellCenterData2d(myg)

    bc = BCObject()

    mydata.register_var("a", bc)
    mydata.create()


    a = mydata.get_var("a")
    a[:,:] = np.exp(-(myg.x2d - 0.5)**2 - (myg.y2d - 1.0)**2)

    print(mydata)

    # output
    print("writing\n")
    mydata.write("mesh_test")

    print("reading\n")
    myg2, myd2 = read("mesh_test")
    print(myd2)


    mydata.pretty_print("a")
