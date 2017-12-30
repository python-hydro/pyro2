Introduction to pyro
====================

.. image:: pyro_plots.png

pyro is a simple framework for implementing and playing with
hydrodynamics solvers.  It is designed to provide a tutorial for
students in computational astrophysics (and hydrodynamics in
general). We introduce simple implementations of some popular methods
used in the field, with the code written to be easily
understandable. It is not parallel, and not meant for demanding
scientific simulations—given the choice between performance and
clarity, clarity is taken.

pyro builds off of a finite-volume framework for solving PDEs. There
are a number of solvers in pyro, allowing for the solution of
hyperbolic (wave), parabolic (diffusion), and elliptic (Poisson)
equations. In particular, the following solvers are developed:

* linear advection

* compressible hydrodynamics

* multigrid

* implicit thermal diffusion

* incompressible hydrodynamics

* low Mach number atmospheric hydrodynamics

Runtime visualization shows the evolution as the equations are solved.

In the pages that follow, the following format is adopted:

* PDF notes provide the basic theory behind the methods.  References
  are cited to provide more detail. 

* An overview of the use of the applicable modules from pyro provided.

* Exercises interspersed fill in some of the detail.


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

* ``util/``: Various service modules used by the pyro routines,
  including runtime parameters, I/O, profiling, and pretty output
  modes.


Fortran
-------

Fortran is used to speed up some critical portions of the code, and in
often cases, provides more clarity than trying to write optimized
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


Running
=======

All the solvers are run through the ``pyro.py`` script. This takes 3
arguments: the solver name, the problem setup to run with that solver
(this is defined in the solver's ``problems/`` sub-directory), and the
inputs file (again, usually from the solver's ``problems/``
directory).

For example, to run the Sedov problem with the compressible solver we would do:

.. code-block:: none

   ./pyro.py compressible sedov inputs.sedov

This knows to look for ``inputs.sedov`` in ``compressible/problems/``
(alternately, you can specify the full path for the inputs file).

To run the smooth Gaussin advection problem with the advection solver, we would do:

.. code-block:: none

   ./pyro.py advection smooth inputs.smooth

Any runtime parameter can also be specified on the command line, after
the inputs file. For example, to disable runtime visualization for the
above run, we could do:

.. code-block:: none

   ./pyro.py advection smooth inputs.smooth vis.dovis=0


Runtime options
---------------

The behavior of the main driver, the solver, and the problem setup can
be controlled by runtime parameters specified in the inputs file (or
via the command line). Runtime parameters are grouped into sections,
with the heading of that section enclosed in ``[ .. ]``. The list of
parameters are stored in the ``pyro/_defaults`` file, the ``_defaults`` files
in the solver directory, and the ``_problem-name.defaults`` file in the
solver's ``problem/`` sub-directory. These three files are parsed at
runtime to define the list of valid parameters. The inputs file is
read next and used to override the default value of any of these
previously defined parameters. Additionally, any parameter can be
specified at the end of the commandline, and these will be used to
override the defaults. The collection of runtime parameters is stored
in a ``RuntimeParameters`` object.

The ``runparams.py`` module in ``util/`` controls access to the runtime
parameters. You can setup the runtime parameters, parse an inputs
file, and access the value of a parameter (``hydro.cfl`` in this example)
as:

.. code-block:: python

   rp = RuntimeParameters()
   rp.load_params("inputs.test")
   ...
   cfl = rp.get_param("hydro.cfl")

When pyro is run, the file ``inputs.auto`` is output containing the
full list of runtime parameters, their value for the simulation, and
the comment that was associated with them from the ``_defaults``
files. This is a useful way to see what parameters are in play for a
given simulation.

All solvers use the following parameters:

+-------------------------------------------------------------------------------------------------------------------------------+
| ``[driver]``                                                                                                                  |
+=====================+=========================================================================================================+
|``max_steps``        | the maximum number of steps in the simulation                                                           |
+---------------------+---------------------------------------------------------------------------------------------------------+
|``tmax``             | the simulation time to evolve to                                                                        |
+---------------------+---------------------------------------------------------------------------------------------------------+
|``init_tstep_factor``| the amount by which to shrink the first timestep. This lets the code ramp up to the CFL timestep slowly |
+---------------------+---------------------------------------------------------------------------------------------------------+
|``max_dt_change``    | the maximum factor by which the timestep can increase from one step to the next                         |
+---------------------+---------------------------------------------------------------------------------------------------------+

+-------------------------------------------------------------------------------------------------------------------------------+
| ``[io]``                                                                                                                      |
+=====================+=========================================================================================================+
|``basename``         | the descriptive prefix to use for output files                                                          |
+---------------------+---------------------------------------------------------------------------------------------------------+
|``dt_out``           | the interval in simulation time between writing output files                                            |
+---------------------+---------------------------------------------------------------------------------------------------------+
|``n_out``            | the number of timesteps between writing output files                                                    |
+---------------------+---------------------------------------------------------------------------------------------------------+

+-------------------------------------------------------------------------------------------------------------------------------+
| ``[vis]``                                                                                                                     |
+=====================+=========================================================================================================+
|``dovis``            | enable (1) or disable (0) runtime visualization                                                         |
+---------------------+---------------------------------------------------------------------------------------------------------+
|``store_images``     | if 1, write out PNG files as we do the runtime visualization                                            |
+---------------------+---------------------------------------------------------------------------------------------------------+
|``n_out``            | the number of timesteps between writing output files                                                    |
+---------------------+---------------------------------------------------------------------------------------------------------+

+-------------------------------------------------------------------------------------------------------------------------------+
| ``[mesh]``                                                                                                                    |
+=====================+=========================================================================================================+
|``xmin``             | the physical coordinate of the lower x face of the domain                                               |
+---------------------+---------------------------------------------------------------------------------------------------------+
|``xmax``             | the physical coordinate of the upper x face of the domain                                               |
+---------------------+---------------------------------------------------------------------------------------------------------+
|``ymin``             | the physical coordinate of the lower y face of the domain                                               |
+---------------------+---------------------------------------------------------------------------------------------------------+
|``ymax``             | the physical coordinate of the upper y face of the domain                                               |
+---------------------+---------------------------------------------------------------------------------------------------------+
|``xlboundary``       | the physical description for the type of boundary at the lower x face of the domain                     |
+---------------------+---------------------------------------------------------------------------------------------------------+
|``xrboundary``       | the physical description for the type of boundary at the upper x face of the domain                     |
+---------------------+---------------------------------------------------------------------------------------------------------+
|``ylboundary``       | the physical description for the type of boundary at the lower y face of the domain                     |
+---------------------+---------------------------------------------------------------------------------------------------------+
|``yrboundary``       | the physical description for the type of boundary at the upper y face of the domain                     |
+---------------------+---------------------------------------------------------------------------------------------------------+
|``nx``               | the number zones in the x-direction                                                                     |
+---------------------+---------------------------------------------------------------------------------------------------------+
|``ny``               | the number zones in the y-direction                                                                     |
+---------------------+---------------------------------------------------------------------------------------------------------+


Working with output
===================

Utilities
---------

Several simply utilities exist to operate on output files

* ``compare.py:`` this script takes two plot files and compares them
  zone-by-zone and reports the differences. This is useful for
  testing, to see if code changes affect the solution. Many problems
  have stored benchmarks in their solver's tests directory. For
  example, to compare the current results for the incompressible shear
  problem to the stored benchmark, we would do:

  .. code-block:: none

    ./compare.py shear_128_0216.pyro incompressible/tests/shear_128_0216.pyro

  Differences on the order of machine precision are may arise because
  of optimizations and compiler differences across platforms. Students
  should familiarize themselves with the details of how computers
  store numbers (floating point). An excellent read is `What every
  computer scientist should know about floating-point arithmetic`
  by D. Goldberg.

* ``plot.py``: this script uses the solver's ``dovis()`` routine to
  plot an output file. For example, to plot the data in the file
  ``shear_128_0216.pyro`` from the incompressible shear problem, you
  would do:

  .. code-block:: none

     ./plot.py -o image.png shear_128_0216.pyro

  where the ``-o`` option allows you to specify the output file name.


Reading and plotting manually
-----------------------------

pyro data can be read using the ``patch.read`` method. The following
sequence (done in a python session) reads in stored data (from the
compressible Sedov problem) and plots data falling on a line in the x
direction through the y-center of the domain (note: this will include
the ghost cells).

.. code-block:: python

   import matplotlib.pyplot as plt
   import util.io as io
   sim = io.read("sedov_unsplit_0000.h5")
   dens = sim.cc_data.get_var("density")
   plt.plot(dens.g.x, dens[:,dens.g.ny//2])
   plt.show()


Adding a problem
================

The easiest way to add a problem is to copy an existing problem setup
in the solver you wish to use (in its problems/ sub-directory. Three
different files will need to be copied (created):

* ``problem.py``: this is the main initialization routine. The
  function ``init_data()`` is called at runtime by the ``Simulation``
  object's ``initialize()`` method. Two arguments are passed in, the
  simulation's ``CellCenterData2d`` object and the
  ``RuntimeParameters`` object.  The job of ``init_data()`` is to fill
  all of the variables defined in the ``CellCenterData2d`` object.

* ``_problem.defaults``: this contains the runtime parameters and
  their defaults for your problem. They should be placed in a block
  with the heading ``[problem]`` (where ``problem`` is your problem's
  name). Anything listed here will be available through the
  ``RuntimeParameters`` object at runtime.

* ``inputs.problem``: this is the inputs file that is used at runtime
  to set the parameters for your problem. Any of the general
  parameters (like the grid size, boundary conditions, etc.) as well
  as the problem-specific parameters can be set here.  Once the
  problem is defined, you need to add the problem name to the
  ``__all__`` list in the ``__init__.py`` file in the ``problems/``
  sub-directory. This lets python know about the problem.

