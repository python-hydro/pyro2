# 4.1

  * Switched to pyproject.toml (#195)

  * pytest improvements allowing it to be run more easily (#194)

  * plotvar.py script improvements (#178)

  * a new viscous Burgers solver was added (#171)

  * a new viscous incompressible solver was added with a lid-drive
    cavity test problem (#138)

  * the incompressible solver was synced up with the Burgers solver
    (#168, #169)

  * convergence.py can now take any variable and multiplicative
    factor, as well as take 3 plotfiles to estimate convergence
    directly. (#165)

  * the multigrid solver output is now more compact (#161)

  * plot.py can fill ghostcells now (#156)

  * a new inviscid Burgers solver was added (#144)

  * a new convergence_error.py script for incompressible was added to
    make the convergence plot for that solver (#147)

  * regression tests can now be run in parallel (#145)

  * fixes for numpy > 1.20 (#137)

  * we can now Ctrl+C to abort when visualization is on (#131)

  * lots of pylint cleaning (#155, #152, #151, #143, #139)

# 4.0

  This begins a new development campaign, with the source updated to
  conform to a standard python packaging format, allowing us to put
  it up on PyPI, and install and run from anywhere.

# 3.1

  This is essentially the version from JOSS

# 3.0

  This was the transition to Numba

# 2.2

  * added shallow water solver

  * added particles support

# 2.1

  * documentation switched to Sphinx

  * extensive flake8 clean-ups

  * travis-ci for unit test (with 79% coverage)

  * SDC compressible method added

# 2.0

  This is the state of the project toward the end of 2017.
  4th order finite-volume methods were just introduced.
