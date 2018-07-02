Design ideas
============

pyro is written primarily in python (by default, we expect python 3),
with a few low-level routines written in Fortran for performance. The
``numpy`` package is used for representing arrays throughout the
python code and the ``matplotlib`` library is used for
visualization. We use ``f2py`` (part of NumPy) to interface with some
Fortran code. Finally, ``pytest`` is used for unit testing of some
components.

All solvers are written for a 2-d grid.  This gives a good balance
between complexity and speed.

A paper describing the design philosophy of pyro was accepted to
Astronomy & Computing `[paper link] <http://adsabs.harvard.edu/abs/2013arXiv1306.6883Z>`_.


Directory structure
-------------------

The files for each solver are in their own sub-directory, with
additional sub-directories for the mesh and utilities. Each solver has
two sub-directories: ``problems/`` and ``tests/``. These store the
different problem setups for the solver and reference output for
testing.

Your ``PYTHONPATH`` environment variable should be set to include the
top-level ``pyro2/`` directory.

The overall structure is:

* ``pyro2/``: This is the top-level directory.  The main driver,
  ``pyro.py``, is here, and all pyro simulations should be run from
  this directory.

* ``advection/``: The linear advection equation solver using the CTU
  method. All advection-specific routines live here.

  * ``problems/``: The problem setups for the advection solver.
  * ``tests/``: Reference advection output files for comparison and regression testing.

* ``advection_rk/``: The linear advection equation solver using the
  method-of-lines approach.

  * ``problems/``: This is a symbolic link to the advection/problems/ directory.
  * ``tests/``: Reference advection output files for comparison and regression testing.

* ``analysis/``: Various analysis scripts for processing pyro output files.

* ``compressible/``: The compressible hydrodynamics solver using the
  CTU method. All source files specific to this solver live here.

  * ``problems/``: The problem setups for the compressible hydrodynamics solver.
  * ``tests/``: Reference compressible hydro output for regression testing.

* ``compressible_rk/``: The compressible hydrodynamics solver using method of lines integration.

  * ``problems/``: This is a symbolic link to the compressible/problems/ directory.
  * ``tests/``: Reference compressible hydro output for regression testing.

* ``diffusion/``: The implicit (thermal) diffusion solver. All diffusion-specific routines live here.

  * ``problems/``: The problem setups for the diffusion solver.
  * ``tests/``: Reference diffusion output for regression testing.

* ``incompressible/``: The incompressible hydrodynamics solver. All incompressible-specific routines live here.

  * ``problems/``: The problem setups for the incompressible solver.
  * ``tests/``:  Reference incompressible hydro output for regression testing.

* ``lm_atm/``: The low Mach number hydrodynamics solver for atmospherical flows. All low-Mach-specific files live here.

  * ``problems/``: The problem setups for the low Mach number solver.
  * ``tests/``: Reference low Mach hydro output for regression testing.

* ``mesh/``: The main classes that deal with 2-d cell-centered grids
  and the data that lives on them. All the solvers use these classes
  to represent their discretized data.

* ``multigrid/``: The multigrid solver for cell-centered data. This
  solver is used on its own to illustrate how multigrid works, and
  directly by the diffusion and incompressible solvers.

  * ``problems/``: The problem setups for when the multigrid solver is used in a stand-alone fashion.
  * ``tests/``: Reference multigrid solver solutions (from when the multigrid solver is used stand-alone) for regression testing.

* ``swe/``: The shallow water solver.

  * ``problems/``: The problem setups for the shallow water solver.
  * ``tests/``: Reference shallow water output for regression testing.

* ``util/``: Various service modules used by the pyro routines,
  including runtime parameters, I/O, profiling, and pretty output
  modes.


Fortran
-------

Fortran is used to speed up some critical portions of the code, and in
many cases, provides more clarity than trying to write optimized
python code using array operations in numpy. The Fortran code
seemlessly integrates into python using f2py.

Wherever Fortran is used, we enforce the following design rule: the
Fortran functions must be completely self-contained, with all
information coming through the interface. No external dependencies
are allowed. Each pyro module will have (at most) a single Fortran
file and can be compiled into a library via a single f2py command line
invocation.

A single script, ``mk.sh``, in the top-level directory will compile
all the Fortran source.


Main driver
-----------

All the solvers use the same driver, the main ``pyro.py`` script. The
flowchart for the driver is:

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
