Adding a problem
================

Problem setups are defined in a ``problems/`` subdirectory under each
solver.  For example, the problem setups for ``compressible`` are here:
https://github.com/python-hydro/pyro2/tree/main/pyro/compressible/problems

When you install pyro via ``pip``, these problem setups become available.
At the moment, the way to add a new problem is to directly put files
into these directories.

.. tip::

   If you are working on adding problems, it is
   recommended that you install pyro from source and do it as an
   `editable install
   <https://setuptools.pypa.io/en/latest/userguide/development_mode.html>`_
   as:

   .. prompt:: bash

      pip install -e .

Every problem needs a python module of the form *problem_name.py*.
This will define the runtime parameters that the problem expects
and do the initialization of the state variables.

Many problems will also provide an *inputs* file that overrides
some of the runtime parameter defaults (like the domain size and BCs)
to be appropriate for this problem.

"`problem.py`"
--------------

A python module named after the problem (we'll call it ``problem.py`` here)
provides the following:

* ``PROBLEM_PARAMS`` : this is a dictionary (it can be empty, ``{}``)
  that defines the runtime parameters that are needed to this problem
  setup.  For instance, for the ``compressible`` ``sod`` problem:

  .. code:: python

     PROBLEM_PARAMS = {"sod.direction": "x",
                       "sod.dens_left": 1.0,
                       "sod.dens_right": 0.125,
                       "sod.u_left": 0.0,
                       "sod.u_right": 0.0,
                       "sod.p_left": 1.0,
                       "sod.p_right": 0.1}

  Each key in the dictionary should be of the form
  *problem-name.parameter*, and the values are the default value of
  the parameter.

  Any of these runtime parameters can be overridden in an inputs file
  or on the commandline (when running via ``pyro_sim.py``) or via the
  ``inputs_dict`` keyword argument (when running via the
  :func:`Pyro <pyro.pyro_sim.Pyro>` class).

* ``DEFAULT_INPUTS`` : this is the name of an inputs file to be
  read in by default when using the ``Pyro`` class interface.  It
  can be ``None``.

  This is not used when running via ``pyro_sim.py``.

* ``init_data()`` : this is the main initialization routine.  It has
  the signature:

  .. code:: python

     def init_data(my_data, rp)

  where

  * ``my_data`` is a :func:`CellCenterData2d <pyro.mesh.patch.CellCenterData2d>` or :func:`FV2d <pyro.mesh.fv.FV2d>` object.  The ``Grid`` object can be obtained from this
    as needed.

  * ``rp`` is a :func:`RuntimeParameters <pyro.util.runparams.RuntimeParameters>` object.
    Any of the runtime parameters (including the problem-specific ones
    defined via ``PROBLEM_PARAMS``) can be accessed via this.

  .. note::

     The interface for ``init_data`` is the same for all solvers.

  The job of ``init_data`` is to initialize the state data that is
  managed by the ``my_data`` object passed in.  Exactly which variables
  are included there will depend on the solver.

* ``finalize()`` : this is called at the very end of evolution.  It
  is meant to output instructions to the user on how the can analyze the
  data.  It takes no arguments.

.. important::

   Once the problem is defined, you need to add the problem name to
   the ``__all__`` list in the ``__init__.py`` file in the
   ``problems/`` sub-directory. This lets python know about the
   problem.
