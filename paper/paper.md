---
title: 'pyro: a framework for hydrodynamics explorations and prototyping'

tags:
- python
- hydrodynamics
- astrophysics
- physics
- partial differential equations

authors:
- name: Michael Zingale
  orcid: 0000-0001-8401-030X
  affiliation: 1
- name: Alice Harpole
  orcid:
  affiliation: 1
- name: Ian Hawke
  orcid:
  affiliation: 2

affiliations:
- name: Department of Physics and Astronomy, Stony Brook University
  index: 1
- name: University of Southampton
  index: 2
  
date: 10 August 2018

bibliography: paper.bib
---

# Summary

pyro

In the time since the first pyro paper [@pyroI], the code has
undergone considerable development, gained a large number of solvers,
adopted unit testing through pytest and documentation through sphinx,
and a number of new contributors.  pyro's functionality can now
be accessed directly through a `Pyro()` class, in addition to the
original commandline script interface.  This new interface in particular
allows for easy use within Jupyter notebooks.  

The original goal of pyro was to learn hydrodynamics methods through
example, and it still serves this goal.  At Stony Brook, pyro is used
with new undergraduate researchers in our group to introduce them to
the ideas of computational hydrodynamics.  But the current framework
has evolved to the point where pyro is used for prototyping
hydrodynamics solvers before implementing them into science codes.  An
example of this is the 4th-order compressible solver built on the ideas
of spectral deferred corrections (the `compressible_sdc` solver).  This
implementation was used as the model for the development of higher-order
schemes in the Castro hydrodynamics code [@castro].  The low Mach
number atmospheric solver (`lm_atm`) is based on the Maestro code [@maestro]
and the pyro implementation will be used to prototype new low Mach number
algorithms before porting them to science codes.

The current pyro solvers are:

  * linear advection (including a second-order unsplit CTU scheme, a
    method-of-lines piecewise linear solver$^\star$, a 4th-order
    finite-volume scheme$^\star$, and a WENO method$^\star$)

  * compressible hydrodynamics (including a second-order unsplit CTU
    scheme, a method-of-lines piecewise linear solver$^\star$,
    and two 4th-order finite-volume schemes, one with Runge-Kutta
    integration and the other using a spectral deferred corrections method$^\star$)

  * diffusion using a second-order discretization

  * incompressible

  * low Mach number atmospheric solver

  * shallow water equations solver

(solvers since the first pyro paper are marked with a $^\star$).  Also
new is support for Lagrangian tracer particles, which can be added to
any solver that has a velocity field.


# References
