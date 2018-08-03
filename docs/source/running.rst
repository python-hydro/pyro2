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

To run the smooth Gaussian advection problem with the advection solver, we would do:

.. code-block:: none

   ./pyro.py advection smooth inputs.smooth

Any runtime parameter can also be specified on the command line, after
the inputs file. For example, to disable runtime visualization for the
above run, we could do:

.. code-block:: none

   ./pyro.py advection smooth inputs.smooth vis.dovis=0


.. note::

   Quite often, the slowest part of the runtime is the visualization, so disabling
   vis as shown above can dramatically speed up the execution.  You can always
   plot the results after the fact using the ``plot.py`` script, as discussed
   in  :ref:`analysis`.

Runtime options
---------------

The behavior of the main driver, the solver, and the problem setup can
be controlled by runtime parameters specified in the inputs file (or
via the command line). Runtime parameters are grouped into sections,
with the heading of that section enclosed in ``[ .. ]``. The list of
parameters are stored in three places:

* the ``pyro/_defaults`` file
* the solver's ``_defaults`` file
* problem's ``_defaults`` file (named ``_problem-name.defaults`` in the
  solver's ``problem/`` sub-directory).

These three files are parsed at runtime to define the list of valid
parameters. The inputs file is read next and used to override the
default value of any of these previously defined
parameters. Additionally, any parameter can be specified at the end of
the commandline, and these will be used to override the defaults. The
collection of runtime parameters is stored in a
:func:`RuntimeParameters <util.runparams.RuntimeParameters>` object.

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
