Design ideas
============

pyro is written entirely in python,
with a few low-level routines compiled *just-in-time* by ``numba`` for performance. The
``numpy`` package is used for representing arrays throughout the
python code and the ``matplotlib`` library is used for
visualization. Finally, ``pytest`` is used for unit testing of some
components.

All solvers are written for a 2-d grid.  This gives a good balance
between complexity and speed.

A paper describing the design philosophy of pyro was published in
Astronomy & Computing `[A&C paper link] <http://adsabs.harvard.edu/abs/2013arXiv1306.6883Z>`_.
A follow-on paper was published in JOSS `[JOSS paper link] <https://joss.theoj.org/papers/10.21105/joss.01265>`_.

Directory structure
-------------------

pyro follows a standard python package structure.  The main directory
(called ``pyro2/`` for historical reasons) contains:

* ``docs/`` : The documentation in Sphinx format

* ``examples/`` : Some example notebooks

* ``paper/`` : the original JOSS paper

* ``presentations/`` : some presentations given on pyro in the past

* ``pyro/`` : the main source directory

* ``www/`` : the logo used in the website

It is at this level (``pyro2/``) that the installation of pyro is done (via ``pyproject.toml``).

``pyro/``
^^^^^^^^^

The main code is all contained in the ``pyro/`` subdirectory.  Here we discuss that.

The files for each solver are in their own sub-directory, with
additional sub-directories for the mesh and utilities. Each solver has
two sub-directories for problems and tests, appearing as:

* *solver-name*

  * ``problems/`` : the problem setups and inputs file that work for this solver.
    In some cases, this might be a symlink to a similar solver that we inherit from.

  * ``tests/``. reference output (HDF5 files) used for the regression testing.

The other directories include:

* ``analysis/``: Various analysis scripts for processing pyro output files.

* ``mesh/``: The main classes that deal with 2-d cell-centered grids
  and the data that lives on them. All the solvers use these classes
  to represent their discretized data.

* ``multigrid/``: The multigrid solver for cell-centered data. This
  solver is used on its own to illustrate how multigrid works, and
  directly by the diffusion and incompressible solvers.

  This includes its own ``problems`` and ``tests`` directories for when
  it is run in a standalone fashion.

* ``particles/``: The solver for Lagrangian tracer particles.  This is meant
  to be used with another solver.

* ``util/``: Various service modules used by the pyro routines,
  including runtime parameters, I/O, profiling, and pretty output
  modes.




Numba
-----

Numba is used to speed up some critical portions of the code. Numba is
a *just-in-time compiler* for python. When a call is first made to a
function decorated with Numba's ``@njit`` decorator, it is compiled to
machine code 'just-in-time' for it to be executed. Once compiled, it
can then run at (near-to) native machine code speed.

We also use Numba's ``cache=True`` option, which means that once the
code is compiled, Numba will write the code into a file-based cache. The next
time you run the same bit of code, Numba will use the saved version rather than
compiling the code again, saving some compilation time at the start of the
simulation.


Main driver
-----------

All the solvers use the same driver, the main ``pyro_sim.py`` script,
contained in ``pyro2/pyro/``. The flowchart for the driver is:

* parse runtime parameters

* setup the grid (``initialize()`` function from the solver)

  * initialize the data for the desired problem (``init_data()`` function from the problem)

* do any necessary pre-evolution initialization (``preevolve()`` function from the solver)

* evolve while t < tmax and n < max_steps

  * fill boundary conditions (``fill_BC_all()`` method of the ``CellCenterData2d`` class)
  * get the timestep (``compute_timestep()`` calls the solver's ``method_compute_timestep()`` function from the solver)
  * evolve for a single timestep (``evolve()`` function from the solver)
  * t = t + dt
  * output (``write()`` method of the ``CellCenterData2d`` class)
  * visualization (``dovis()`` function from the solver)

* call the solver's ``finalize()`` function to output any useful information at the end

This format is flexible enough for the advection, compressible,
diffusion, and incompressible evolution solver. Each solver provides a
``Simulation`` class that provides the following methods (note:
inheritance is used, so many of these methods come from the base
``NullSimulation`` class):

* ``compute_timestep``: return the timestep based on the solver's
  specific needs (through ``method_compute_timestep()``) and
  timestepping parameters in the driver

* ``dovis``: performs visualization of the current solution

* ``evolve``: advances the system of equations through a single timestep

* ``finalize``: any final clean-ups, printing of analysis hints.

* ``finished``: return True if we've met the stopping criteria for a simulation

* ``initialize``: sets up the grid and solution variables

* ``method_compute_timestep``: returns the timestep for evolving the system

* ``preevolve``: does any initialization to the fluid state that is necessary before the main evolution. Not every solver will need something here.

* ``read_extras``: read in any solver-specific data from a stored output file

* ``write``: write the state of the simulation to an HDF5 file

* ``write_extras``: any solver-specific writing

Each problem setup needs only provide an ``init_data()`` function that fills the data in the patch object.
