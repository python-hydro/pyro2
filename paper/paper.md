---
title: 'pyro: a framework for hydrodynamics explorations and prototyping'

tags:
- Python
- hydrodynamics
- astrophysics
- physics
- partial differential equations

authors:
- name: Alice Harpole
  orcid: 0000-0002-1530-781X
  affiliation: 1
- name: Michael Zingale
  orcid: 0000-0001-8401-030X
  affiliation: 1
- name: Ian Hawke
  orcid: 0000-0003-4805-0309
  affiliation: 2
- name: Taher Chegini
  orcid: 0000-0002-5430-6000
  affiliation: 3
affiliations:
- name: Department of Physics and Astronomy, Stony Brook University
  index: 1
- name: University of Southampton
  index: 2
- name: University of Houston
  index: 3

date: 10 August 2018

bibliography: paper.bib
---

# Summary

`pyro` is a Python-based simulation framework designed for ease of
implementation and exploration of hydrodynamics methods.  It is
built in a object-oriented fashion, allowing for the reuse of
the core components and fast prototyping of new methods.

The original goal of `pyro` was to learn hydrodynamics methods through
example, and it still serves this goal.  At Stony Brook, `pyro` is used
with new undergraduate researchers in our group to introduce them to
the ideas of computational hydrodynamics.  But the current framework
has evolved to the point where `pyro` is used for prototyping
hydrodynamics solvers before implementing them into science codes.  An
example of this is the 4th-order compressible solver built on the
ideas of spectral deferred corrections (the `compressible_sdc`
solver).  This implementation was used as the model for the
development of higher-order schemes in the Castro hydrodynamics code
[@castro].  The low Mach number atmospheric solver (`lm_atm`) is based
on the Maestro code [@maestro] and the `pyro` implementation will be
used to prototype new low Mach number algorithms before porting them
to science codes.

In the time since the first `pyro` paper [@pyroI], the code has
undergone considerable development, gained a large number of solvers,
adopted unit testing through pytest and documentation through sphinx,
and a number of new contributors.  `pyro`'s functionality can now
be accessed directly through a `Pyro()` class, in addition to the
original commandline script interface.  This new interface in particular
allows for easy use within Jupyter notebooks.  We also now use HDF5
for output instead of Python's `pickle()` function.  Previously, we used Fortran
to speed up some performance-critical portions of the code.  These routines
could be called by the main Python code by first compiling them using `f2py`.
In the new version, we have replaced these Fortran routines by Python functions
that are compiled at runtime by `numba`.  Consequently, `pyro` is now written
entirely in Python.

The current `pyro` solvers are:

-   linear advection (including a second-order unsplit CTU scheme, a
    method-of-lines piecewise linear solver$^\star$, a 4th-order
    finite-volume scheme$^\star$, a WENO method$^\star$, and
    advection with a non-uniform velocity field$^\star$)

-   compressible hydrodynamics (including a second-order unsplit CTU
    scheme, a method-of-lines piecewise linear solver$^\star$, and two
    4th-order finite-volume schemes, one with Runge-Kutta integration
    and the other using a spectral deferred corrections
    method$^\star$)

-   diffusion using a second-order implicit discretization

-   incompressible hydrodynamics using a second-order approximate
    projection method.

-   low Mach number atmospheric solver$^\star$, using an approximate
    projection method.

-   shallow water equations solver$^\star$

(solvers since the first `pyro` paper are marked with a $^\star$).  Also,
new is support for Lagrangian tracer particles, which can be added to
any solver that has a velocity field.

# Acknowledgements

The work at Stony Brook was supported by DOE/Office of Nuclear Physics
grant DE-FG02-87ER40317 and DOE grant DE-SC0017955.

# References
