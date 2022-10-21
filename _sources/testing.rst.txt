Testing
=======

There are two types of testing implemented in pyro: unit tests and
regression tests.  Both of these are driven by the ``test.py``
script in the root directory.


Unit tests
----------

pyro implements unit tests using ``py.test``.  These can be run via:

.. prompt:: bash

   ./test.py -u


Regression tests
----------------

The main driver, ``pyro_sim.py`` has the ability to create benchmarks and
compare output to stored benchmarks at the end of a simulation.
Benchmark output is stored in each solver's ``tests/`` directory.
When testing, we compare zone-by-zone for each variable to see if we
agree exactly.  If there is any disagreement, this means that we've
made a change to the code that we need to understand (if may be a bug
or may be a fix or optimization).

We can compare to the stored benchmarks simply by running:

.. prompt:: bash

   ./test.py


.. note::

   When running on a new machine, it is possible that roundoff-level differences
   may mean that we do not pass the regression tests.  In this case, one would
   need to create a new set of benchmarks for that machine and use those for
   future tests.
