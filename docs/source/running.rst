Running
=======

Pyro can be run in two ways:

* from the commandline, using the ``pyro_sim.py`` script (this will be installed into your search path)

* by using the :func:`Pyro <pyro.pyro_sim.Pyro>` class directly in ipython or Jupyter

Commandline
------------

The ``pyro_sim.py`` script takes 3
arguments: the solver name, the problem setup to run with that solver
(this is defined in the solver's ``problems/`` sub-directory), and the
inputs file (again, usually from the solver's ``problems/``
directory).

For example, to run the Sedov problem with the compressible solver we would do:

.. prompt:: bash

   pyro_sim.py compressible sedov inputs.sedov

This knows to look for ``inputs.sedov`` in ``compressible/problems/``
(alternately, you can specify the full path for the inputs file).

To run the smooth Gaussian advection problem with the advection solver, we would do:

.. prompt:: bash

   pyro_sim.py advection smooth inputs.smooth

Any runtime parameter can also be specified on the command line, after
the inputs file. For example, to disable runtime visualization for the
above run, we could do:

.. prompt:: bash

   pyro_sim.py advection smooth inputs.smooth vis.dovis=0


.. note::

   Quite often, the slowest part of the runtime is the visualization, so disabling
   vis as shown above can dramatically speed up the execution.  You can always
   plot the results after the fact using the ``plot.py`` script, as discussed
   in  :ref:`analysis`.


Pyro class
----------

Alternatively, pyro can be run using the :func:`Pyro <pyro.pyro_sim.Pyro>` class. This provides
an interface that enables simulations to be set up and run in a Jupyter notebook -- see
``examples/examples.ipynb`` for an example notebook. A simulation can be set up and run
by carrying out the following steps:

* create a :func:`Pyro <pyro.pyro_sim.Pyro>` object, initializing it with a specific solver
* initialize the problem, passing in runtime parameters and inputs
* run the simulation

For example, if we wished to use the :mod:`compressible <pyro.compressible>` solver to run the
Kelvin-Helmholtz problem ``kh``, we would do the following:

.. code-block:: python

    from pyro import Pyro
    p = Pyro("compressible")
    p.initialize_problem(problem_name="kh")
    p.run_sim()

This will use the default set of parameters for the problem specified
in the inputs file defined by ``DEFAULT_INPUTS`` in the problem
initialization module.

Instead of using an inputs file to define the problem parameters, we can define a
dictionary of parameters and pass them into the :func:`initialize_problem
<pyro.pyro_sim.Pyro.initialize_problem>` function using the keyword argument ``inputs_dict``.
If an inputs file is also passed into the function, the parameters in the dictionary
will override any parameters in the file. For example, if we wished to turn off
visualization for the previous example, we would do:

.. code-block:: python

    parameters = {"vis.dovis": 0}
    p.initialize_problem(problem_name="kh",
                         inputs_dict=parameters)

It's possible to evolve the simulation forward timestep by timestep manually using
the :func:`single_step <pyro.pyro_sim.Pyro.single_step>` function (rather than allowing
:func:`run_sim <pyro.pyro_sim.Pyro.run_sim>` to do this for us). To evolve our example
simulation forward by a single step, we'd run

.. code-block:: python

    p.single_step()

This will fill the boundary conditions, compute the timestep ``dt``, evolve a
single timestep and do output/visualization (if required).


Runtime options
---------------

The behavior of the main driver, the solver, and the problem setup can
be controlled by runtime parameters specified in the inputs file (or
via the command line or passed into the ``initialize_problem`` function).
Runtime parameters are grouped into sections,
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
:func:`RuntimeParameters <pyro.util.runparams.RuntimeParameters>` object.

The ``runparams.py`` module in ``util/`` controls access to the runtime
parameters. You can setup the runtime parameters, parse an inputs
file, and access the value of a parameter (``hydro.cfl`` in this example)
as:

.. code-block:: python

   rp = RuntimeParameters()
   rp.load_params("inputs.test")
   ...
   cfl = rp.get_param("hydro.cfl")

.. tip::

   When pyro is run, the file ``inputs.auto`` is output containing the
   full list of runtime parameters, their value for the simulation, and
   the comment that was associated with them from the ``_defaults``
   files. This is a useful way to see what parameters are in play for a
   given simulation.

All solvers use the following parameters:

.. include:: _defaults.inc
