Mesh overview
=============

All solvers are based on a finite-volume/cell-centered
discretization. The basic theory of such methods is discussed in
:ref:`notes`.

.. note::

   The core data structure that holds data on the grid is
   :func:`CellCenterData2d <mesh.patch.CellCenterData2d>`.  This does
   not distinguish between cell-centered data and cell-averages.  This
   is fine for methods that are second-order accurate, but for
   higher-order methods, the :func:`FV2d <mesh.fv.FV2d>` class has
   methods for converting between the two data centerings.


``mesh.patch`` implementation and use
-------------------------------------

We import the basic mesh functionality as:

.. code-block:: python

   import mesh.patch as patch
   import mesh.fv as fv
   import mesh.boundary as bnd
   import mesh.array_indexer as ai

There are several main objects in the patch class that we interact with:

* :func:`patch.Grid2d <mesh.patch.Grid2d>`: this is the main grid
  object. It is basically a container that holds the number of zones
  in each coordinate direction, the domain extrema, and the
  coordinates of the zones themselves (both at the edges and center).

* :func:`patch.CellCenterData2d <mesh.patch.CellCenterData2d>`: this
  is the main data object—it holds cell-centered data on a grid.  To
  build a ``CellCenterData2d`` object you need to pass in the ``Grid2d``
  object that defines the mesh. The ``CellCenterData2d`` object then
  allocates storage for the unknowns that live on the grid. This class
  also provides methods to fill boundary conditions, retrieve the data
  in different fashions, and read and write the object from/to disk.

* :func:`fv.FV2d <mesh.fv.FV2d>`: this is a special class derived from
  ``patch.CellCenterData2d`` that implements some extra functions
  needed to convert between cell-center data and averages with
  fourth-order accuracy.

* :func:`bnd.BC <mesh.boundary.BC>`: This is simply a container that
  holds the names of the boundary conditions on each edge of the
  domain.

* :func:`ai.ArrayIndexer <mesh.array_indexer.ArrayIndexer>`: This is a
  class that subclasses the NumPy ndarray and makes the data in the
  array know about the details of the grid it is defined on. In
  particular, it knows which cells are valid and which are the ghost
  cells, and it has methods to do the :math:`a_{i+1,j}` operations that are
  common in difference methods.

* :func:`integration.RKIntegrator <mesh.integration.RKIntegrator>`:
  This class implements Runge-Kutta integration in time by managing a
  hierarchy of grids at different time-levels.  A Butcher tableau
  provides the weights and evaluation points for the different stages
  that make up the integration.

The procedure for setting up a grid and the data that lives on it is as follows:

.. code-block:: python

   myg = patch.Grid2d(16, 32, xmax=1.0, ymax=2.0)

This creates the 2-d grid object ``myg`` with 16 zones in the x-direction
and 32 zones in the y-direction. It also specifies the physical
coordinate of the rightmost edge in x and y.

.. code-block:: python

   mydata = patch.CellCenterData2d(myg)

   bc = bnd.BC(xlb="periodic", xrb="periodic", ylb="reflect-even", yrb="outflow")

   mydata.register_var("a", bc)
   mydata.create()


This creates the cell-centered data object, ``mydata``, that lives on the
grid we just built above. Next we create a boundary condition object,
specifying the type of boundary conditions for each edge of the
domain, and finally use this to register a variable, ``a`` that lives on
the grid. Once we call the ``create()`` method, the storage for the
variables is allocated and we can no longer add variables to the grid.
Note that each variable needs to specify a BC—this allows us to do
different actions for each variable (for example, some may do even
reflection while others may do odd reflection).

Jupyter notebook
----------------

A Jupyter notebook that illustrates some of the basics of working with
the grid is provided as :ref:`mesh-examples.ipynb`. This will
demonstrate, for example, how to use the ``ArrayIndexer`` methods to
construct differences.


Tests
-----

The actual filling of the boundary conditions is done by the ``fill_BC()``
method. The script ``bc_demo.py`` tests the various types of boundary
conditions by initializing a small grid with sequential data, filling
the BCs, and printing out the results.
