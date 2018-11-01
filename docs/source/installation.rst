Setting up pyro
===============

You can clone pyro from github: `http://github.com/python-hydro/pyro2 <http://github.com/python-hydro/pyro2>`_

.. note::

   It is strongly recommended that you use python 3.x.  While python 2.x might
   still work, we do not test pyro under python 2, so it may break at any time
   in the future.

The following python packages are required:

* ``numpy``
* ``matplotlib``
* ``numba``
* ``pytest`` (for unit tests)

By default, ``pyro`` will run using code written entirely in python (with the help
of ``numba`` to speed up some functions). However, some of the functions also have
Fortran versions which run several times faster than their python counterparts.
This can be useful for running more intensive calculations (e.g. with higher
resolution or a large number of timesteps). To compile the Fortran functions,
you will need the python package ``f2py`` (part of NumPy) and have ``gFortran``
installed on your computer for ``f2py`` to be able to compile the code.

The following steps are needed before running pyro:

* add ``pyro/`` to your ``PYTHONPATH`` environment variable.  For
  the bash shell, this is done as:

    .. code-block:: none

       export PYTHONPATH="/path/to/pyro/:${PYTHONPATH}"

* build the modules by running the ``mk.sh`` script. To use the python-only version, it
  should be sufficient to just do:

    .. code-block:: none

       ./mk.sh


  Alternatively, to build the faster Fortran modules do:

    .. code-block:: none

       ./mk.sh fortran

* define the environment variable ``PYRO_HOME`` to point to
  the ``pyro2/`` directory

    .. code-block:: none

       export PYRO_HOME="/path/to/pyro/"

.. note::

    If you have built the Fortran modules and want to go back to using the python-only
    modules, you will need to first run ``./mk.sh clean``.

Quick test
----------

Run the advection solver to quickly test if things are setup correctly:

.. code-block:: none

   ./pyro.py advection smooth inputs.smooth

You should see a plot window pop up with a smooth pulse advecting
diagonally through the periodic domain.
