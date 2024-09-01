Adding a problem
================

The easiest way to add a problem is to copy an existing problem setup
in the solver you wish to use (in its problems/ sub-directory). Three
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
