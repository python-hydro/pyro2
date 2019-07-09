"""
The patch module defines the classes necessary to describe finite-volume
data and the grid that it lives on.

Typical usage:

* create the grid::

   grid = Domain2d(nx, ny)

* create the data that lives on that grid::

   data = ParticleData2d(grid)

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

import h5py

from util import msg

from sph.compute import compute_density

class Variables(object):
    """
    a container class for easy access to the different sph
    variables by an integer key
    """
    def __init__(self, myd):
        self.nvar = len(myd.names)

        # conserved variables -- we set these when we initialize for
        # they match the CellCenterData2d object
        self.im = myd.names.index("mass")
        self.irho = myd.names.index("density")
        self.ix = myd.names.index("x-position")
        self.iy = myd.names.index("y-position")
        self.iu = myd.names.index("x-velocity")
        self.iv = myd.names.index("y-velocity")
        self.iuh = myd.names.index("half-x-velocity")
        self.ivh = myd.names.index("half-y-velocity")
        self.iax = myd.names.index("x-acceleration")
        self.iay = myd.names.index("y-acceleration")

class Domain2d(object):
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

    def __init__(self, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0):
        """
        Create a Domain2d object.

        The only data that we require is the number of points that
        make up the mesh in each direction.  Optionally we take the
        extrema of the domain (default is [0,1]x[0,1]) and number of
        ghost cells (default is 1).

        Note that the Domain2d object only defines the discretization,
        it does not know about the boundary conditions, as these can
        vary depending on the variable.

        Parameters
        ----------
        xmin : float, optional
            Physical coordinate at the lower x boundary
        xmax : float, optional
            Physical coordinate at the upper x boundary
        ymin : float, optional
            Physical coordinate at the lower y boundary
        ymax : float, optional
            Physical coordinate at the upper y boundary
        """

        # domain extrema
        self.xmin = xmin
        self.xmax = xmax

        self.ymin = ymin
        self.ymax = ymax

    def __eq__(self, other):
        """ are two domains equivalent? """
        result = (self.xmin == other.xmin and self.xmax == other.xmax and
                  self.ymin == other.ymin and self.ymax == other.ymax)

        return result

class ParticleData2d(object):
    """
    A class to define cell-centered data that lives on a grid.  A
    ParticleData2d object is built in a multi-step process before
    it can be used.

    * Create the object.  We pass in a grid object to describe where
      the data lives::

         my_data = patch.ParticleData2d(myGrid)

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

    def __init__(self, domain, rp, bc, dtype=np.float64):

        """
        Initialize the ParticleData2d object.

        Parameters
        ----------
        domain : Domain2d object
            The domain upon which the data will live
        dtype : NumPy data type, optional
            The datatype of the data we wish to create (defaults to
            np.float64
        """

        self.domain = domain
        self.np = rp.get_param("sph.np")

        self.dtype = dtype
        self.data = None

        self.names = []
        self.vars = self.names  # backwards compatibility hack
        self.nvar = 0

        self.aux = {}

        self.BCs = bc

        # time
        self.t = -1.0

        self.initialized = 0

    def register_var(self, name):
        """
        Register a variable with ParticleData2d object.

        Parameters
        ----------
        name : str
            The variable name
        bc : BC object
            The boundary conditions that describe the actions to take
            for this variable at the physical domain boundaries.
        """

        if self.initialized == 1:
            msg.fail("ERROR: domain already initialized")

        self.names.append(name)
        self.nvar += 1

    def set_aux(self, keyword, value):
        """
        Set any auxillary (scalar) data.  This data is simply carried
        along with the ParticleData2d object

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
            msg.fail("ERROR: domain already initialized")

        self.data = np.zeros((self.np, self.nvar))

        self.initialized = 1

    def __str__(self):
        """ print out some basic information about the ParticleData2d
            object """

        if self.initialized == 0:
            my_str = "ParticleData2d object not yet initialized"
            return my_str

        my_str = "         nvars = {}\n".format(self.nvar)
        my_str += "         variables:\n"

        for n in range(self.nvar):
            my_str += "%16s: min: %15.10f    max: %15.10f\n" % \
                (self.names[n], self.min(self.names[n]), self.max(self.names[n]))
            # my_str += "%16s  BCs: -x: %-12s +x: %-12s -y: %-12s +y: %-12s\n" %\
            #     (" ", self.BCs[self.names[n]].xlb,
            #           self.BCs[self.names[n]].xrb,
            #           self.BCs[self.names[n]].ylb,
            #           self.BCs[self.names[n]].yrb)

        return my_str

    def get_var(self, name):
        """
        Return a data array for the variable described by name.  Stored
        variables will be checked first, and then any derived variables
        will be checked.

        For a stored variable, changes made to this are automatically
        reflected in the ParticleData2d object.

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
        except ValueError:
            for f in self.derives:
                var = f(self, name)
                if len(var) > 0:
                    return var
            raise KeyError("name {} is not valid".format(name))
        else:
            return self.data[:, n]

    def get_var_by_index(self, n):
        """
        Return a data array for the variable with index n in the
        data array.  Any changes made to this are automatically
        reflected in the ParticleData2d object.

        Parameters
        ----------
        n : int
            The index of the variable to access

        Returns
        -------
        out : ndarray
            The array of data corresponding to the index

        """
        return self.data[:, n]

    def get_vars(self):
        """
        Return the entire data array.  Any changes made to this
        are automatically reflected in the ParticleData2d object.

        Returns
        -------
        out : ndarray
            The array of data

        """
        return self.data

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
        self.data[:, n] = 0.0

    def fill_BC_all(self):
        """
        Fill boundary conditions on all variables.
        """
        ivars = Variables(self)

        bc = self.BCs
        U = self.data

        # -x boundary
        if bc.xlb in ["outflow", "neumann"]:
            # in the future we could just kill these particles
            pass
        elif bc.xlb in ["reflect-even", "reflect-odd"]:
            # print("hi i'm lower x reflective")
            # tbounce = (U[:, ivars.ix] - self.domain.xmin) / U[:, ivars.iu]
            for p in range(self.np):
                if U[p, ivars.ix] < self.domain.xmin:
                    U[p, ivars.ix] = 2 * self.domain.xmin - U[p, ivars.ix]
                    U[p, ivars.iu] = -U[p, ivars.iu]
                    U[p, ivars.iuh] = -U[p, ivars.iuh]
        elif bc.xlb == "periodic":
            # print("hi i'm lower x periodic")
            for p in range(self.np):
                if U[p, ivars.ix] < self.domain.xmin:
                    U[p, ivars.ix] = self.domain.xmax - (self.domain.xmin - U[p, ivars.ix])

        # +x boundary
        if bc.xrb in ["outflow", "neumann"]:
            # in the future we could just kill these particles
            pass
        elif bc.xrb in ["reflect-even", "reflect-odd"]:
            for p in range(self.np):
                if U[p, ivars.ix] > self.domain.xmax:
                    U[p, ivars.ix] = 2 * self.domain.xmax - U[p, ivars.ix]
                    U[p, ivars.iu] = -U[p, ivars.iu]
                    U[p, ivars.iuh] = -U[p, ivars.iuh]
        elif bc.xrb == "periodic":
            for p in range(self.np):
                if U[p, ivars.ix] > self.domain.xmax:
                    U[p, ivars.ix] = self.domain.xmin + (U[p, ivars.ix] - self.domain.xmax)

        # -y boundary
        if bc.ylb in ["outflow", "neumann"]:
            # in the future we could just kill these particles
            pass
        elif bc.ylb in ["reflect-even", "reflect-odd"]:
            for p in range(self.np):
                if U[p, ivars.iy] < self.domain.ymin:
                    U[p, ivars.iy] = 2 * self.domain.ymin - U[p, ivars.iy]
                    U[p, ivars.iv] = -U[p, ivars.iv]
                    U[p, ivars.ivh] = -U[p, ivars.ivh]
        elif bc.ylb == "periodic":
            for p in range(self.np):
                if U[p, ivars.iy] < self.domain.ymin:
                    U[p, ivars.iy] = self.domain.ymax - (self.domain.ymin - U[p, ivars.iy])

        # +y boundary
        if bc.yrb in ["outflow", "neumann"]:
            # in the future we could just kill these particles
            pass
        elif bc.yrb in ["reflect-even", "reflect-odd"]:
            for p in range(self.np):
                if U[p, ivars.iy] > self.domain.ymax:
                    U[p, ivars.iy] = 2 * self.domain.ymax - U[p, ivars.iy]
                    U[p, ivars.iv] = -U[p, ivars.iv]
                    U[p, ivars.ivh] = -U[p, ivars.ivh]
        elif bc.yrb == "periodic":
            for p in range(self.np):
                if U[p, ivars.iy] > self.domain.ymax:
                    U[p, ivars.iy] = self.domain.ymin + (U[p, ivars.iy] - self.domain.ymax)


    def min(self, name):
        """
        return the minimum of the variable name in the domain's valid region
        """
        n = self.names.index(name)
        return np.min(self.data[:, n])

    def max(self, name):
        """
        return the maximum of the variable name in the domain's valid region
        """
        n = self.names.index(name)
        return np.max(self.data[:, n])

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

        # data
        gstate = f.create_group("state")

        # for n in range(self.nvar):
        #     gvar = gstate.create_group(self.names[n])
        #     gvar.create_dataset("data",
        #                         data=self.get_var_by_index(n).v())
        #     gvar.attrs["xlb"] = self.BCs[self.names[n]].xlb
        #     gvar.attrs["xrb"] = self.BCs[self.names[n]].xrb
        #     gvar.attrs["ylb"] = self.BCs[self.names[n]].ylb
        #     gvar.attrs["yrb"] = self.BCs[self.names[n]].yrb

    def pretty_print(self, var, fmt=None):
        """print out the contents of the data array with pretty formatting
        indicating where ghost cells are."""
        a = self.get_var(var)
        a.pretty_print(fmt=fmt)

    def normalize_mass(self):
        ivars = Variables(self)

        U = self.data

        compute_density(self, ivars)

        rho0 = self.get_aux("rho0")
        rho2s = np.sum(U[:, ivars.irho]**2)
        rhos = np.sum(U[:, ivars.irho])

        U[:, ivars.im] *= rho0 * rhos / rho2s

def particle_data_clone(old):
    """
    Create a new ParticleData2d object that is a copy of an existing
    one

    Parameters
    ----------
    old : ParticleData2d object
        The ParticleData2d object we wish to copy

    Note
    ----
    It may be that this whole thing can be replaced with a copy.deepcopy()

    """

    if not isinstance(old, ParticleData2d):
        msg.fail("Can't clone object")

    # we may be a type derived from ParticleData2d, so use the same
    # type
    myt = type(old)
    new = myt(old.domain, old.np, dtype=old.dtype)

    for n in range(old.nvar):
        new.register_var(old.names[n], old.BCs[old.names[n]])

    new.create()

    new.aux = old.aux.copy()
    new.data = old.data.copy()

    return new


def do_demo():
    """ show examples of the patch methods / classes """

    pass

    # import util.io as io
    #
    # # illustrate basic mesh operations
    #
    # myg = Domain2d(8, 16, xmax=1.0, ymax=2.0)
    #
    # mydata = ParticleData2d(myg)
    #
    # bc = bnd.BC()
    #
    # mydata.register_var("a", bc)
    # mydata.create()
    #
    # a = mydata.get_var("a")
    # a[:, :] = np.exp(-(myg.x2d - 0.5)**2 - (myg.y2d - 1.0)**2)
    #
    # print(mydata)
    #
    # # output
    # print("writing\n")
    # mydata.write("mesh_test")
    #
    # print("reading\n")
    # myd2 = io.read("mesh_test")
    # print(myd2)
    #
    # mydata.pretty_print("a")


if __name__ == "__main__":
    do_demo()
