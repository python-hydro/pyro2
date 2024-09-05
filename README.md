[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI Version](https://img.shields.io/pypi/v/pyro-hydro)](https://pypi.org/project/pyro-hydro)
[![pylint](https://github.com/python-hydro/pyro2/actions/workflows/pylint.yml/badge.svg)](https://github.com/python-hydro/pyro2/actions/workflows/pylint.yml)
[![flake8](https://github.com/python-hydro/pyro2/actions/workflows/flake8.yml/badge.svg)](https://github.com/python-hydro/pyro2/actions/workflows/flake8.yml)
[![pytest-all](https://github.com/python-hydro/pyro2/actions/workflows/pytest.yml/badge.svg)](https://github.com/python-hydro/pyro2/actions/workflows/pytest.yml)
[![github pages](https://github.com/python-hydro/pyro2/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/python-hydro/pyro2/actions/workflows/gh-pages.yml)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/python-hydro/pyro2/main?filepath=examples%2Fexamples.ipynb)

[![JOSS status](http://joss.theoj.org/papers/6d8b2f94e6d08a7b5d65e98a948dcad7/status.svg)](http://joss.theoj.org/papers/6d8b2f94e6d08a7b5d65e98a948dcad7)
[![DOI](https://zenodo.org/badge/20570861.svg)](https://zenodo.org/badge/latestdoi/20570861)




![pyro logo](www/logo.gif)

*A simple python-based tutorial on computational methods for hydrodynamics*

pyro is a computational hydrodynamics code that presents
two-dimensional solvers for advection, compressible hydrodynamics,
diffusion, incompressible hydrodynamics, and multigrid, all in a
finite-volume framework.  The code is mainly written in python and is
designed with simplicity in mind.  The algorithms are written to
encourage experimentation and allow for self-learning of these code
methods.

The latest version of pyro is always available at:

https://github.com/python-hydro/pyro2

The project webpage, where you'll find documentation, plots, notes,
etc. is here:

https://python-hydro.github.io/pyro2

(Note: there is an outdated page at readthedocs.org that is no longer updated)


## Table of Contents

 * [Getting started](#getting-started)
 * [Core Data Structures](#core-data-structures)
 * [Solvers](#solvers)
 * [Working with data](#working-with-data)
 * [Understanding the algorithms](#understanding-the-algorithms)
 * [Regression and unit testing](#regression-and-unit-testing)
 * [Acknowledgements](#acknowledgements)
 * [Getting help](#getting-help)


## Getting started

  - By default, we assume python 3.9 or later.

  - We require `numpy`, `numba`, `matplotlib`, and `h5py` for running pyro
    and `setuptools_scm` for the install.

  - There are several ways to install pyro.  The simplest way is via
    PyPI/pip:

    ```
    pip install pyro-hydro
    ```

    Alternately, you can install directly from source, via

    ```
    pip install .
    ```

    you can optionally add the `-e` argument to `install` if you are
    planning on developing pyro solvers directly.

  - Not all matplotlib backends allow for the interactive plotting as
    pyro is run. One that does is the TkAgg backend. This can be made
    the default by creating a file `~/.matplotlib/matplotlibrc` with
    the content:

       `backend: TkAgg`

    You can check what backend is your current default in python via:

      ```python
      import matplotlib.pyplot
      print matplotlib.pyplot.get_backend()
      ```

 - If you want to run the unit tests, you need to have `pytest`
   installed.

 - Finally, you can run a quick test of the advection solver:

   ```
   pyro_sim.py advection smooth inputs.smooth
   ```

   you should see a graphing window pop up with a smooth pulse
   advecting diagonally through the periodic domain.


## Core Data Structures

The main data structures that describe the grid and the data the lives
on the grid are described in a jupyter notebook:

https://github.com/python-hydro/pyro2/blob/main/pyro/mesh/mesh-examples.ipynb


Many of the methods here rely on multigrid.  The basic multigrid
solver is demonstrated in the juputer notebook:

https://github.com/python-hydro/pyro2/blob/main/pyro/multigrid/multigrid-constant-coefficients.ipynb


## Solvers

pyro provides the following solvers (all in 2-d):

  - `advection`: a second-order unsplit linear advection solver.  This
    uses characteristic tracing and corner coupling for the prediction
    of the interface states.  This is the basic method to understand
    hydrodynamics.

  - `advection_fv4`: a fourth-order accurate finite-volume advection
    solver that uses RK4 time integration.

  - `advection_nonuniform`: a solver for advection with a non-uniform
    velocity field.

  - `advection_rk`: a second-order unsplit solver for linear advection
    that uses Runge-Kutta integration instead of characteristic
    tracing.

  - `advection_weno`: a method-of-lines WENO solver for linear
    advection.

  - `burgers`: a second-order unsplit solver for invsicid Burgers'
    equation.

  - `viscous_burgers`: a second-order unsplit solver for viscous
    Burgers' equation with constant-coefficient diffusion. It uses
    Crank-Nicolson time-discretized solver for solving diffusion.

  - `compressible`: a second-order unsplit solver for the Euler
    equations of compressible hydrodynamics.  This uses characteristic
    tracing and corner coupling for the prediction of the interface
    states and a 2-shock or HLLC approximate Riemann solver.

  - `compressible_fv4`: a fourth-order accurate finite-volume
    compressible hydro solver that uses RK4 time integration.  This
    is built from the method of McCourquodale and Colella (2011).

  - `compressible_rk`: a second-order unsplit solver for Euler
    equations that uses Runge-Kutta integration instead of
    characteristic tracing.

  - `compressible_sdc`: a fourth-order compressible solver, using
    spectral-deferred correction (SDC) for the time integration.

  - `diffusion`: a Crank-Nicolson time-discretized solver for the
    constant-coefficient diffusion equation.

  - `incompressible`: a second-order cell-centered approximate
    projection method for the incompressible equations of
    hydrodynamics.

  - `incompressible_viscous`: an extension of the incompressible solver
    including a diffusion term for viscosity.

  - `lm_atm`: a solver for the equations of low Mach number
    hydrodynamics for atmospheric flows.

  - `lm_combustion`: (in development) a solver for the equations of
    low Mach number hydrodynamics for smallscale combustion.

  - `multigrid`: a cell-centered multigrid solver for a
    constant-coefficient Helmholtz equation, as well as a
    variable-coefficient Poisson equation (which inherits from the
    constant-coefficient solver).

  - `particles`: a solver for Lagrangian tracer particles.

  - `swe`: a solver for the shallow water equations.


## Working with data

In addition to the main pyro program, there are many analysis tools
that we describe here. Note: some problems write a report at the end
of the simulation specifying the analysis routines that can be used
with their data.

  - `pyro/util/compare.py`: this takes two simulation output files as
    input and compares zone-by-zone for exact agreement. This is used
    as part of the regression testing.

      usage: `./compare.py file1 file2`

  - `pyro/plot.py`: this takes the an output file as input and plots
    the data using the solver's dovis method.

      usage: `./plot.py file`

  - `pyro/analysis/`

      * `dam_compare.py`: this takes an output file from the shallow
        water dam break problem and plots a slice through the domain
        together with the analytic solution (calculated in the
        script).

         usage: `./dam_compare.py file`

      * `gauss_diffusion_compare.py`: this is for the diffusion
        solver's Gaussian diffusion problem. It takes a sequence of
        output files as arguments, computes the angle-average, and the
        plots the resulting points over the analytic solution for
        comparison with the exact result.

          usage: `./gauss_diffusion_compare.py file*`

      * `incomp_converge_error.py`: this is for the incompressible
        solver's converge problem. This takes a single output file as
        input and compares the velocity field to the analytic
        solution, reporting the L2 norm of the error.

          usage: `./incomp_converge_error.py file`

      * `plotvar.py`: this takes a single output file and a variable
        name and plots the data for that variable.

          usage: `./plotvar.py file variable`

      * `sedov_compare.py`: this takes an output file from the
        compressible Sedov problem, computes the angle-average profile
        of the solution and plots it together with the analytic data
        (read in from `cylindrical-sedov.out`).

          usage: `./sedov_compare.py file`

      * `smooth_error.py`: this takes an output file from the
        advection solver's smooth problem and compares to the analytic
        solution, outputting the L2 norm of the error.

          usage: `./smooth_error.py file`

      * `sod_compare.py`: this takes an output file from the
        compressible Sod problem and plots a slice through the domain
        over the analytic solution (read in from `sod-exact.out`).

          usage: `./sod_compare.py file`


## Understanding the algorithms

  There is a set of notes that describe the background and details of
  the algorithms that pyro implements:

  http://open-astrophysics-bookshelf.github.io/numerical_exercises/

  The source for these notes is also available on github:

  https://github.com/Open-Astrophysics-Bookshelf/numerical_exercises


## Regression and unit testing

  The `pyro/test.py` script will run several of the problems (as well
  as some stand-alone multigrid tests) and compare the solution to
  stored benchmarks (in each solver's `tests/` subdirectory).  The
  return value of the script is the number of tests that failed.

  Unit tests are controlled by pytest and can be run simply via

  ```
  pytest
  ```

## Acknowledgements

  If you use pyro in a class or workshop, please e-mail us to let us
  know (we'd like to start listing these on the website).

  If pyro was used for a publication, please cite the article found in
  the `CITATION` file.


## Getting help

  We use github discussions as a way to ask about the code:

  https://github.com/python-hydro/pyro2/discussions
