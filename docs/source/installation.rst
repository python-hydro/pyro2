Setting up pyro
===============

You can clone pyro from github: `http://github.com/python-hydro/pyro2 <http://github.com/python-hydro/pyro2>`_

The following python packages are required:

* ``numpy``
* ``matplotlib``
* ``numba``
* ``h5py``
* ``pytest`` (for unit tests)

The easiest way to install python is to use the ``setup.py``.  From
the ``pyro2/`` directory, we do:

.. prompt:: bash

   python setup.py install --user

This will put the main driver, ``pyro_sim.py``, in your path, and
allow you to run pyro from anywhere.

If you intend on directly developing the solvers, you can instead do:

.. prompt:: bash

   python setup.py develop --user

This will allow you to modify the python source without having to
reinstall each time something changes.


Quick test
----------

Run the advection solver to quickly test if things are setup correctly:

.. code-block:: none

   ./pyro.py advection smooth inputs.smooth

You should see a plot window pop up with a smooth pulse advecting
diagonally through the periodic domain.
