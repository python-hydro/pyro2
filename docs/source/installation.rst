Setting up pyro
===============

You can clone pyro from github: `http://github.com/zingale/pyro2 <http://github.com/zingale/pyro2>`_

.. note::

   It is strongly recommended that you use python 3.x.  While python 2.x might
   still work, we do not test pyro under python 2, so it may break at any time
   in the future.

The following python packages are required:

* ``numpy``
* ``matplotlib``
* ``f2py`` (part of NumPy)
* ``pytest`` (for unit tests)

Additionally, you will need ``gfortran`` installed on your computer, for
``f2py`` to be able to compile the code.

The following steps are needed before running pyro:

* add ``pyro/`` to your ``PYTHONPATH`` environment variable.  For
  the bash shell, this is done as:

.. code-block:: none

   export PYTHONPATH="/path/to/pyro/:${PYTHONPATH}"

* build the Fortran modules, by running the ``mk.sh`` script. It
  should be sufficient to just do:

.. code-block:: none

   ./mk.sh

* define the environment variable ``PYRO_HOME`` to point to
  the ``pyro2/`` directory (only needed for regression testing)

.. code-block:: none

   export PYRO_HOME="/path/to/pyro/"


Quick test
----------

Run the advection solver to quickly test if things are setup correctly:

.. code-block:: none

   ./pyro.py advection smooth inputs.smooth

You should see a plot window pop up with a smooth pulse advecting
diagonally through the periodic domain.
