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

# keep track of whether the BCs are solid walls (passed into the
# Riemann solver).  
bc_props = {}
bc_props["outflow"] = False
bc_props["periodic"] = False
bc_props["reflect"] = True
bc_props["reflect-even"] = True
bc_props["reflect-odd"] = True
bc_props["dirichlet"] = True
bc_props["neumann"] = False

extBCs = {}

def define_bc(type, function, is_solid=False):
    """
    use this to extend the types of boundary conditions supported
    on a solver-by-solver basis.  Here we pass in the reference to
    a function that can be called with the data that needs to be
    filled.  is_solid indicates whether it should be interpreted as
    a solid wall (no flux through the BC)"
    """

    bc_props[type] = is_solid
    extBCs[type] = function


def _set_reflect(odd_reflect_dir, dir_string):
    if odd_reflect_dir == dir_string:
        return "reflect-odd"
    else:
        return "reflect-even"


class BCObject(object):
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

    def __init__ (self,
                  xlb="outflow", xrb="outflow",
                  ylb="outflow", yrb="outflow",
                  xl_func=None, xr_func=None,
                  yl_func=None, yr_func=None,
                  grid=None,
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
        
        valid = list(bc_props.keys())

        # -x boundary
        if xlb in valid:
            self.xlb = xlb
            if self.xlb == "reflect":
                self.xlb = _set_reflect(odd_reflect_dir, "x")
        else:
            msg.fail("ERROR: xlb = %s invalid BC" % (xlb))

        # +x boundary
        if xrb in valid:
            self.xrb = xrb
            if self.xrb == "reflect":
                self.xrb = _set_reflect(odd_reflect_dir, "x")
        else:
            msg.fail("ERROR: xrb = %s invalid BC" % (xrb))

        # -y boundary
        if ylb in valid:
            self.ylb = ylb
            if self.ylb == "reflect":
                self.ylb = _set_reflect(odd_reflect_dir, "y")
        else:
            msg.fail("ERROR: ylb = %s invalid BC" % (ylb))

        # +y boundary
        if yrb in valid:
            self.yrb = yrb
            if self.yrb == "reflect":
                self.yrb = _set_reflect(odd_reflect_dir, "y")
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
        self.xl_value = self.xr_value = self.yl_value = self.yr_value = None
        
        if not xl_func == None:
            self.xl_value = xl_func(grid.y)
        if not xr_func == None:
            self.xr_value = xr_func(grid.y)
        if not yl_func == None:
            self.yl_value = yl_func(grid.x)
        if not yr_func == None:
            self.yr_value = yr_func(grid.x)

    def __str__(self):
        """ print out some basic information about the BC object """

        string = "BCs: -x: %s  +x: %s  -y: %s  +y: %s" % \
            (self.xlb, self.xrb, self.ylb, self.yrb)

        return string


def _buf_split(b):
    try: bxlo, bxhi, bylo, byhi = b
    except:
        try: blo, bhi = b
        except:
            blo = b
            bhi = b
        bxlo = bylo = blo
        bxhi = byhi = bhi
    return bxlo, bxhi, bylo, byhi


class ArrayIndexer(object):
    """ a class that wraps the data region of a single array (d)
        and allows us to easily do array operations like d[i+1,j]
        using the ip() method. """


    # ?? Can we accomplish this a lot easier by subclassing
    # the ndarray?
    # e.g, the InfoArray example here:
    # http://docs.scipy.org/doc/numpy/user/basics.subclassing.html    
    def __init__(self, d=None, grid=None):
        self.d = d
        self.g = grid
        s = d.shape
        self.c = len(s)
        
    def __add__(self, other):
        if isinstance(other, ArrayIndexer):        
            return ArrayIndexer(d=self.d + other.d, grid=self.g)
        else:
            return ArrayIndexer(d=self.d + other, grid=self.g)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, ArrayIndexer):        
            return ArrayIndexer(d=self.d - other.d, grid=self.g)
        else:
            return ArrayIndexer(d=self.d - other, grid=self.g)            

    def __mul__(self, other):
        if isinstance(other, ArrayIndexer):
            return ArrayIndexer(d=self.d * other.d, grid=self.g)
        else:
            return ArrayIndexer(d=self.d * other, grid=self.g)            

    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __truediv__(self, other):
        if isinstance(other, ArrayIndexer):        
            return ArrayIndexer(d=self.d / other.d, grid=self.g)
        else:
            return ArrayIndexer(d=self.d / other, grid=self.g)

    def __div__(self, other):
        if isinstance(other, ArrayIndexer):        
            return ArrayIndexer(d=self.d / other.d, grid=self.g)
        else:
            return ArrayIndexer(d=self.d / other, grid=self.g)            

    def __rdiv__(self, other):
        if isinstance(other, ArrayIndexer):        
            return ArrayIndexer(d=other.d / self.d, grid=self.g)
        else:
            return ArrayIndexer(d=other / self.d, grid=self.g)            

    def __rtruediv__(self, other):
        if isinstance(other, ArrayIndexer):        
            return ArrayIndexer(d=other.d / self.d, grid=self.g)
        else:
            return ArrayIndexer(d=other / self.d, grid=self.g)            
        
    def __pow__(self, other):
        return ArrayIndexer(d=self.d**2, grid=self.g)

    def __abs__(self):
        return ArrayIndexer(d=np.abs(self.d), grid=self.g)    
    
    def v(self, buf=0, n=0, s=1):
        return self.ip_jp(0, 0, buf=buf, n=n, s=s)
        
    def ip(self, shift, buf=0, n=0, s=1):
        return self.ip_jp(shift, 0, buf=buf, n=n, s=s)
        
    def jp(self, shift, buf=0, n=0, s=1):
        return self.ip_jp(0, shift, buf=buf, n=n, s=s)

    def ip_jp(self, ishift, jshift, buf=0, n=0, s=1):
        bxlo, bxhi, bylo, byhi = _buf_split(buf)

        if self.c == 2:
            return self.d[self.g.ilo-bxlo+ishift:self.g.ihi+1+bxhi+ishift:s,
                          self.g.jlo-bylo+jshift:self.g.jhi+1+byhi+jshift:s]
        else:
            return self.d[self.g.ilo-bxlo+ishift:self.g.ihi+1+bxhi+ishift:s,
                          self.g.jlo-bylo+jshift:self.g.jhi+1+byhi+jshift:s,n]

    def norm(self, n=0):
        """
        find the norm of the quantity (index n) defined on the same grid,
        in the domain's valid region

        """
        if self.c == 2:
            return self.g.norm(self.d)
        else:
            return self.g.norm(self.d[:,:,n])
                
    def sqrt(self):
        return ArrayIndexer(d=np.sqrt(self.d), grid=self.g)

    def min(self):
        return self.d.min()

    def max(self):
        return self.d.max()    

    def copy(self):
        return ArrayIndexer(d=self.d.copy(), grid=self.g)

    def is_symmetric(self, nodal=False, tol=1.e-14):
        if not nodal:
            L = self.d[self.g.ilo:self.g.ilo+self.g.nx/2,
                       self.g.jlo:self.g.jhi+1]
            R = self.d[self.g.ilo+self.g.nx/2:self.g.ihi+1,
                       self.g.jlo:self.g.jhi+1]
        else:
            print(self.g.ilo,self.g.ilo+self.g.nx/2+1)
            L = self.d[self.g.ilo:self.g.ilo+self.g.nx/2+1,
                       self.g.jlo:self.g.jhi+1]
            print(self.g.ilo+self.g.nx/2,self.g.ihi+2)
            R = self.d[self.g.ilo+self.g.nx/2:self.g.ihi+2,
                       self.g.jlo:self.g.jhi+1]


        e = abs(L - np.flipud(R)).max()
        print(e, tol, e < tol)
        return e < tol


    def is_asymmetric(self, nodal=False, tol=1.e-14):
        if not nodal:
            L = self.d[self.g.ilo:self.g.ilo+self.g.nx/2,
                       self.g.jlo:self.g.jhi+1]
            R = self.d[self.g.ilo+self.g.nx/2:self.g.ihi+1,
                       self.g.jlo:self.g.jhi+1]
        else:
            print(self.g.ilo,self.g.ilo+self.g.nx/2+1)
            L = self.d[self.g.ilo:self.g.ilo+self.g.nx/2+1,
                       self.g.jlo:self.g.jhi+1]
            print(self.g.ilo+self.g.nx/2,self.g.ihi+2)
            R = self.d[self.g.ilo+self.g.nx/2:self.g.ihi+2,
                       self.g.jlo:self.g.jhi+1]


        e = abs(L + np.flipud(R)).max()
        print(e, tol, e < tol)
        return e < tol


    def pretty_print(self):
        """
        Print out a small dataset to the screen with the ghost cells
        a different color, to make things stand out
        """

        if self.d.dtype == np.int:
            fmt = "%4d"
        elif self.d.dtype == np.float64:
            fmt = "%10.5g"
        else:
            msg.fail("ERROR: dtype not supported")

        # print j descending, so it looks like a grid (y increasing
        # with height)
        for j in reversed(range(self.g.qy)):
            for i in range(self.g.qx):

                if (j < self.g.jlo or j > self.g.jhi or
                    i < self.g.ilo or i > self.g.ihi):
                    gc = 1
                else:
                    gc = 0

                if gc:
                    print("\033[31m" + fmt % (self.d[i,j]) + "\033[0m", end="")
                else:
                    print (fmt % (self.d[i,j]), end="")

            print(" ")

        leg = """
         ^ y
         |
         +---> x
        """
        print(leg)

    
    
class Grid2d():
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

        # center of the grid (for convenience)
        self.ic = self.ilo + nx/2 - 1
        self.jc = self.jlo + ny/2 - 1

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
        return Grid2d(self.nx/N, self.ny/N, ng=self.ng,
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
        return "2-d grid: nx = {}, ny = {}, ng = {}".format(self.nx, self.ny,self.ng)


    def __eq__(self, other):
        """ are two grids equivalent? """
        result = (self.nx == other.nx and self.ny == other.ny and 
                  self.ng == other.ng and 
                  self.xmin == other.xmin and self.xmax == other.xmax and 
                  self.ymin == other.ymin and self.ymax == other.ymax)

        return result


class CellCenterData2d():
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

        for n in range(self.nvar):
            myStr += "%16s: min: %15.10f    max: %15.10f\n" % \
                (self.vars[n],
                 np.min(self.data[n,ilo:ihi+1,jlo:jhi+1]),
                 np.max(self.data[n,ilo:ihi+1,jlo:jhi+1]) )
            myStr += "%16s  BCs: -x: %-12s +x: %-12s -y: %-12s +y: %-12s\n" %\
                (" " , self.BCs[self.vars[n]].xlb,
                       self.BCs[self.vars[n]].xrb,
                       self.BCs[self.vars[n]].ylb,
                       self.BCs[self.vars[n]].yrb)

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
        return ArrayIndexer(d=self.data[n,:,:], grid=self.grid)


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
        return ArrayIndexer(d=self.data[n,:,:], grid=self.grid)


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

            if self.BCs[name].xl_value == None:            
                for i in range(self.grid.ilo):
                    self.data[n,i,:] = self.data[n,self.grid.ilo,:]
            else:
                self.data[n,self.grid.ilo-1,:] = \
                    self.data[n,self.grid.ilo,:] - self.grid.dx*self.BCs[name].xl_value[:]
                
        elif self.BCs[name].xlb == "reflect-even":

            for i in range(self.grid.ilo):
                self.data[n,i,:] = self.data[n,2*self.grid.ng-i-1,:]

        elif self.BCs[name].xlb in ["reflect-odd", "dirichlet"]:

            if self.BCs[name].xl_value == None:
                for i in range(self.grid.ilo):
                    self.data[n,i,:] = -self.data[n,2*self.grid.ng-i-1,:]
            else:
                self.data[n,self.grid.ilo-1,:] = \
                    2*self.BCs[name].xl_value[:] - self.data[n,self.grid.ilo,:]

        elif self.BCs[name].xlb == "periodic":

            for i in range(self.grid.ilo):
                self.data[n,i,:] = self.data[n,self.grid.ihi-self.grid.ng+i+1,:]


        # +x boundary
        if self.BCs[name].xrb in ["outflow", "neumann"]:

            if self.BCs[name].xr_value == None:
                for i in range(self.grid.ihi+1, self.grid.nx+2*self.grid.ng):
                    self.data[n,i,:] = self.data[n,self.grid.ihi,:]
            else:
                self.data[n,self.grid.ihi+1,:] = \
                    self.data[n,self.grid.ihi,:] + self.grid.dx*self.BCs[name].xr_value[:]

        elif self.BCs[name].xrb == "reflect-even":

            for i in range(self.grid.ng):
                i_bnd = self.grid.ihi+1+i
                i_src = self.grid.ihi-i

                self.data[n,i_bnd,:] = self.data[n,i_src,:]

        elif self.BCs[name].xrb in ["reflect-odd", "dirichlet"]:

            if self.BCs[name].xr_value == None:
                for i in range(self.grid.ng):
                    i_bnd = self.grid.ihi+1+i
                    i_src = self.grid.ihi-i

                    self.data[n,i_bnd,:] = -self.data[n,i_src,:]
            else:
                self.data[n,self.grid.ihi+1,:] = \
                    2*self.BCs[name].xr_value[:] - self.data[n,self.grid.ihi,:]

        elif self.BCs[name].xrb == "periodic":

            for i in range(self.grid.ihi+1, 2*self.grid.ng + self.grid.nx):
                self.data[n,i,:] = self.data[n,i-self.grid.ihi-1+self.grid.ng,:]


        # -y boundary
        if self.BCs[name].ylb in ["outflow", "neumann"]:

            if self.BCs[name].yl_value == None:
                for j in range(self.grid.jlo):
                    self.data[n,:,j] = self.data[n,:,self.grid.jlo]
            else:
                self.data[n,:,self.grid.jlo-1] = \
                    self.data[n,:,self.grid.jlo] - self.grid.dy*self.BCs[name].yl_value[:]

        elif self.BCs[name].ylb == "reflect-even":

            for j in range(self.grid.jlo):
                self.data[n,:,j] = self.data[n,:,2*self.grid.ng-j-1]

        elif self.BCs[name].ylb in ["reflect-odd", "dirichlet"]:

            if self.BCs[name].yl_value == None:
                for j in range(self.grid.jlo):
                    self.data[n,:,j] = -self.data[n,:,2*self.grid.ng-j-1]
            else:
                self.data[n,:,self.grid.jlo-1] = \
                    2*self.BCs[name].yl_value[:] - self.data[n,:,self.grid.jlo]
                
        elif self.BCs[name].ylb == "periodic":

            for j in range(self.grid.jlo):
                self.data[n,:,j] = self.data[n,:,self.grid.jhi-self.grid.ng+j+1]

        else:
            if self.BCs[name].ylb in extBCs.keys():

                extBCs[self.BCs[name].ylb](self.BCs[name].ylb, "ylb", name, self)


        # +y boundary
        if self.BCs[name].yrb in ["outflow", "neumann"]:

            if self.BCs[name].yr_value == None:
                for j in range(self.grid.jhi+1, self.grid.ny+2*self.grid.ng):
                    self.data[n,:,j] = self.data[n,:,self.grid.jhi]
            else:
                self.data[n,:,self.grid.jhi+1] = \
                    self.data[n,:,self.grid.jhi] + self.grid.dy*self.BCs[name].yr_value[:]

        elif self.BCs[name].yrb == "reflect-even":

            for j in range(self.grid.ng):
                j_bnd = self.grid.jhi+1+j
                j_src = self.grid.jhi-j

                self.data[n,:,j_bnd] = self.data[n,:,j_src]

        elif self.BCs[name].yrb in ["reflect-odd", "dirichlet"]:

            if self.BCs[name].yr_value == None:
                for j in range(self.grid.ng):
                    j_bnd = self.grid.jhi+1+j
                    j_src = self.grid.jhi-j

                    self.data[n,:,j_bnd] = -self.data[n,:,j_src]
            else:
                self.data[n,:,self.grid.jhi+1] = \
                    2*self.BCs[name].yr_value[:] - self.data[n,:,self.grid.jhi]
                
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
        cG = fG.coarse_like(2)
        cData = cG.scratch_array()

        # fill the coarse array with the restricted data -- just
        # average the 4 fine cells into the corresponding coarse cell
        # that encompasses them.
        cData.v()[:,:] = \
            0.25*(fData.v(s=2) + fData.ip(1, s=2) +
                  fData.jp(1, s=2) + fData.ip_jp(1, 1, s=2))

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

        # allocate an array for the finely gridded data
        fG = cG.fine_like(2)
        fData = fG.scratch_array()

        # slopes for the coarse data
        m_x = cG.scratch_array()
        m_x.v()[:,:] = 0.5*(cData.ip(1) - cData.ip(-1))

        m_y = cG.scratch_array()
        m_y.v()[:,:] = 0.5*(cData.jp(1) - cData.jp(-1))

        # fill the children
        fData.v(s=2)[:,:] = cData.v() - 0.25*m_x.v() - 0.25*m_y.v()     # 1 child
        fData.ip(1, s=2)[:,:] = cData.v() + 0.25*m_x.v() - 0.25*m_y.v() # 2 
        fData.jp(1, s=2)[:,:] = cData.v() - 0.25*m_x.v() + 0.25*m_y.v() # 3
        fData.ip_jp(1, 1, s=2)[:,:] = cData.v() + 0.25*m_x.v() + 0.25*m_y.v() # 4

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


    def pretty_print(self, var):
        
        a = self.get_var(var)
        a.pretty_print()


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

    for n in range(old.nvar):
        new.register_var(old.vars[n], old.BCs[old.vars[n]])

    new.create()

    new.aux = old.aux.copy()
    new.data = old.data.copy()

    return new


if __name__== "__main__":

    # illustrate basic mesh operations

    myg = Grid2d(8,16, xmax=1.0, ymax=2.0)

    mydata = CellCenterData2d(myg)

    bc = BCObject()

    mydata.register_var("a", bc)
    mydata.create()


    a = mydata.get_var("a")
    a.d[:,:] = np.exp(-(myg.x2d - 0.5)**2 - (myg.y2d - 1.0)**2)

    print(mydata)

    # output
    print("writing\n")
    mydata.write("mesh_test")

    print("reading\n")
    myg2, myd2 = read("mesh_test")
    print(myd2)


    mydata.pretty_print("a")
