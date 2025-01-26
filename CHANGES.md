# 4.5.0

  * implement a "clean state" routine for the compressible solvers
    that can enforce a density floor (#329)

  * updated the citation information (#324) and the zenodo file (#325)

  * code cleaning (#323)

  * added python 3.13 support (#321)

  * CI doc build fixes (#322) and added the ability to skip executing
    notebooks when building docs (#319)

  * fixed issues with how derived variables are recognized when
    reading in an output file and add vorticity and machnumber
    to compressible derives (#294, #320)

  * expand the documentation on the compressible solvers (#317)

  * remove unused APIs from docs (#318) and fix autodoc issues (#316)

  * implement a sponge term in the compressible solvers (#313, #336)

  * add some asserts to the compressible solver to catch unphysical
    mass and energy (#312)

  * add the compressible convection problem (#293, #310, #315, #337)

  * add a multimode Rayleigh-Taylor problem to the compressible
    solvers (#307)

  * improve the documentation on working with data (#304, #305)

  * spherical compressible solver fixes (#295, #296, #298, #300)

  * move the 4th order reconstruction into mesh (#302)

  * use a safer divide in compressible when density is zero (#297)

  * add the ability to have a problem-dependent external source and
    clean-up how the compressible sources are treated (#289, #290,
    #292)

# 4.4.0

  * the gresho problem (compressible) now takes Mach number as input
    instead of pressure (#287, #288)

  * doc updates to reflect how we run with the Pyro class (#266, #276,
    #284, #285) and other general docs cleaning (#265, #268, #270,
    #271)

  * pylint CI update (#283); python version bump in actions (#281)

  * a new HLLC solver with corrections for low Mach number was added
    (based on Minioshima & Miyoshi 2021) (#282)

  * documenting and clean-ups of problem setups for each of the
    solvers (adding docstrings that appear in the docs) (#272, #273,
    #274, #275, #277, #279, #280)

  * `viscous_burgers` was renamed `burgers_viscous` (#267, #278)

# 4.3.0

  * it is now possible to define a problem setup in a Jupyter
    notebook (#262, #264)

  * fix a bug in the artificial viscosity for spherical coords (#263)

  * I/O is disabled by default when running in Jupyter (#259)

  * the bc_demo test works again (#260, #261)

  * problem setups no longer check if the input grid is
    CellCenterData2d (#258)

  * the Pyro class interface was simplified to have command line
    parameter use the dict interface (#257)

  * problem setups no longer use a _*.defaults file, but instead
    specify their runtime parameters via a dict in the problem module
    (#255)

  * the compressible_sr solver was removed (#226)

  * gresho problem uses more steps by default now (#254)

  * the 4th order compressible solver only needs 4 ghost cells, not 5
    (#248)

  * the compressible solver comparison docs were changed to an
    interactive Jupyter page (#243, #246, #249, #252)

  * some interfaces were cleaned-up to require keyword args (#247)

  * developers were added to the zenodo file (#242)

  * doc updates (#241)

# 4.2.0

  * moved docs to sphinx-book-theme (#229, #235, #240)

  * docs reorganization (#214, #221, #222, #234, #239) and new
    examples (#228, #236)

  * remove driver.splitting unused parameter (#238)

  * clean-ups of the Pyro class (#219, #232, #233) including disabling
    verbosity and vis by default when using it directly (#220, #231)

  * the `advection_fv4` solver now properly averages the initial
    conditions from centers to cell-averages

  * each problem initial conditions file now specifies a
    `DEFAULT_INPUTS` (#225)

  * the gresho initial conditions were fixed to be closer to the
    source of the Miczek et al. paper (#218)

  * the RT initial conditions were tweaked to be more symmetric (#216)

  * the colorbar tick labels in plots were fixed (#212)

  * the compressible solver now supports 2D spherical geometry
    (r-theta) (#204, #210, #211)

  * the mesh now supports spherical geometry (r-theta) (#201, #217)

  * the compressible Riemann solvers were reorganized (#206)

  * CI fixes (#202, #215) and a codespell action (#199, #205)

  * python 3.12 was added to the CI (#208)

  * comment fixes to the compressible FV4 solver (#207)

# 4.1.0

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
