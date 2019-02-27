:mod:`mesh.mapped <mesh.mapped>` implementation and use
-------------------------------------------------------

This module defines the structures required for mapped grids. It defines the classes
:func:`MappedGrid2d <mesh.mapped.MappedGrid2d>` and
:func:`MappedCellCenterData2d <mesh.mapped.MappedCellCenterData2d>` that inherit from
the un-mapped versions, :func:`Grid2d <mesh.patch.Grid2d>` and
:func:`CellCenterData2d <mesh.patch.CellCenterData2d>`.

We import the basic mapped grids functionality as:

.. code-block:: python

    import mesh.mapped as mapped

The classes in the mapped module that we interact with are:

* :func:`mapped.MappedGrid2d <mesh.mapped.MappedGrid2d>`: this is the main mapped
  grid class. It holds the coordinate grid (a :func:`Grid2d <mesh.patch.Grid2d>`
  object) and details about the mapping from the physical grid to the
  coordinate grid. The mapping is defined using a sympy function: when the object
  is instantiated, this analytic expression with then be used to calculate
  the various numerical quantities on the actual grid.

* :func:`mapped.MappedCellCenterData2d <mesh.mapped.MappedCellCenterData2d>`: this
  holds the cell-centered data on the coordinate grid. It is identical to its
  parent class, :func:`CellCenterData2d <mesh.patch.CellCenterData2d>`, except
  it also defines the rotation matrices that define how the velocity
  components are mapped between the physical and coordinate grids (these
  rotation matrices cannot be fully defined on the grid itself as they require
  information about the variables which the grid alone does not have).

The procedure for setting up a mapped grid and the data that lives on it is as
follows:

.. code-block:: python

    import sympy
    from sympy.abs import x, y
    import mesh.mapped as mapped
    import mesh.boundary as bnd
    import compressible

    def mapping(myg):
        return sympy.Matrix([2 * x + 3, y**2])

    myg = mapped.MappedGrid2d(mapping, 16, 32, xmax=1.0, ymax=2.0)

This creates a 2-d mapped grid object ``myg`` with 16 zones in the x-direction
and 32 zones in the y-direction on the coordinate grid. It also specifies the
extent of the coordinate grid in the x- and y-directions. The mapping is defined
using a ``sympy.Matrix`` object as a function of the cartesian coordinates
``x`` and ``y``. The mapping itself is a function of the mapped grid object.

.. note::

    The mapping must be specified as a function of the ``sympy.Symbol`` objects
    ``x`` and ``y`` only.

.. code-block:: python

    mydata = mapped.MappedCellCenterData2d(myg)

    bc = bnd.BC(xlb="periodic", xrb="periodic", ylb="reflect-even", yrb="outflow")

    mydata.register_var("a", bc)
    mydata.create()

    ivars = compressible.Variables(mydata)

    mydata.make_rotation_matrices(ivars)

This create the mapped cell-centered data object, ``mydata``, that lives on the
mapped grid we just defined. Just as we would for the non-mapped grid, we next
create a boundary condition object to specify the boundary conditions on each edge,
register variables and call the ``create()`` function to allocate storage for the
variables. For the mapped grid, there is one more step that we must do: define the
rotation matrices associated with the variables. To do this, we define a
``Variables`` object (here we use the one from the compressible module), then
pass this to the data object's :func:`make_rotation_matrices <mesh.mapped.MappedCellCenterData2d.make_rotation_matrices>` method. This will
take the rotation matrix function defined on the mapped grid object ``myg`` and
create the actual matrices associated with the data's variables.
