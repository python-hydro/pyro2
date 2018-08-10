.. _analysis:

Analysis routines
=================

In addition to the main pyro program, there are many analysis tools
that we describe here. Note: some problems write a report at the end
of the simulation specifying the analysis routines that can be used
with their data.

* ``compare.py``: this takes two simulation output files as input and
  compares zone-by-zone for exact agreement. This is used as part of
  the regression testing.

  usage: ``./compare.py file1 file2``

* ``plot.py``: this takes an output file as input and plots the data
  using the solver's dovis method. It deduces the solver from the
  attributes stored in the HDF5 file.

  usage: ``./plot.py file``

* ``analysis/``

  * ``convergence.py``: this compares two files with different
    resolutions (one a factor of 2 finer than the other).  It coarsens
    the finer data and then computes the norm of the difference.  This
    is used to test the convergence of solvers.

  * ``dam_compare.py``: this takes an output file from the
    shallow water dam break problem and plots a slice through the domain
    together with the analytic solution (calculated in the script).

    usage: ``./dam_compare.py file``

  * ``gauss_diffusion_compare.py``: this is for the diffusion solver's
    Gaussian diffusion problem. It takes a sequence of output files as
    arguments, computes the angle-average, and the plots the resulting
    points over the analytic solution for comparison with the exact
    result.

    usage: ``./gauss_diffusion_compare.py file*``

  * ``incomp_converge_error.py``: this is for the incompressible
    solver's converge problem. This takes a single output file as
    input and compares the velocity field to the analytic solution,
    reporting the L2 norm of the error.

    usage: ``./incomp_converge_error.py file``

  * ``plotvar.py``: this takes a single output file and a variable
    name and plots the data for that variable.

    usage: ``./plotvar.py file variable``

  * ``sedov_compare.py``: this takes an output file from the
    compressible Sedov problem, computes the angle-average profile of
    the solution and plots it together with the analytic data (read in
    from ``cylindrical-sedov.out``).

    usage: ``./sedov_compare.py file``

  * ``smooth_error.py``: this takes an output file from the advection
    solver's smooth problem and compares to the analytic solution,
    outputing the L2 norm of the error.

    usage: ``./smooth_error.py file``

  * ``sod_compare.py``: this takes an output file from the
    compressible Sod problem and plots a slice through the domain over
    the analytic solution (read in from ``sod-exact.out``).

    usage: ``./sod_compare.py file``
