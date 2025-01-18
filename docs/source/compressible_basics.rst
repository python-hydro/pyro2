**************************
Compressible hydrodynamics
**************************

The Euler equations of compressible hydrodynamics take the form:

.. math::

   \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho U) &= 0 \\
   \frac{\partial (\rho U)}{\partial t} + \nabla \cdot (\rho U U) + \nabla p &= \rho g \\
   \frac{\partial (\rho E)}{\partial t} + \nabla \cdot [(\rho E + p ) U] &= \rho U \cdot g

with :math:`\rho E = \rho e + \frac{1}{2} \rho |U|^2` and :math:`p =
p(\rho, e)`.  Note these do not include any dissipation terms, since
they are usually negligible in astrophysics.

pyro has several compressible solvers to solve this equation set.
The implementations here have flattening at shocks, artificial
viscosity, a simple gamma-law equation of state, and (in some cases) a
choice of Riemann solvers. Optional constant gravity in the vertical
direction is allowed.

.. note::

   All the compressible solvers share the same ``problems/``
   directory, which lives in ``compressible/problems/``.  For the
   other compressible solvers, we simply use a symbolic-link to this
   directory in the solver's directory.

``compressible`` solver
=======================

:py:mod:`pyro.compressible` is based on a directionally unsplit (the corner
transport upwind algorithm) piecewise linear method for the Euler
equations, following :cite:`colella:1990`.  This is overall second-order
accurate.

The parameters for this solver are:

.. include:: compressible_defaults.inc

.. include:: compressible_problems.inc


``compressible_rk`` solver
==========================

:py:mod:`pyro.compressible_rk` uses a method of lines time-integration
approach with piecewise linear spatial reconstruction for the Euler
equations.  This is overall second-order accurate.

The parameters for this solver are:

.. include:: compressible_rk_defaults.inc

.. include:: compressible_rk_problems.inc

``compressible_fv4`` solver
===========================

:py:mod:`pyro.compressible_fv4` uses a 4th order accurate method with RK4
time integration, following :cite:`mccorquodalecolella`.

The parameter for this solver are:

.. include:: compressible_fv4_defaults.inc

.. include:: compressible_fv4_problems.inc

``compressible_sdc`` solver
===========================

:py:mod:`pyro.compressible_sdc` uses a 4th order accurate method with
spectral-deferred correction (SDC) for the time integration.  This
shares much in common with the :py:mod:`pyro.compressible_fv4` solver, aside from
how the time-integration is handled.

The parameters for this solver are:

.. include:: compressible_sdc_defaults.inc

.. include:: compressible_sdc_problems.inc


.. toctree::
   :maxdepth: 1
   :hidden:

   compressible_problems
   compressible_sources
   compressible_exercises
