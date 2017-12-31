Setting up pyro
===============

The following python packages are requires:

* ``numpy``
* ``matplotlib``
* ``f2py`` (part of NumPy)
* ``pytest`` (for unit tests)

The following steps are needed before running pyro:

* add ``pyro/`` to your ``PYTHONPATH`` environment variable.  For
  the bash shell, this is done as:

.. code-block:: none

   export PYTHONPATH="/path/to/pyro/:${PYTHONPATH}"

* build the Fortran modules, by running the ``mk.sh`` script. It
  should be sufficient to just do:

.. code-block:: none

   ./mk.sh


