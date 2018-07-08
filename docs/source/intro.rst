Introduction to pyro
====================

.. image:: pyro_plots.png

pyro is a simple framework for implementing and playing with
hydrodynamics solvers.  It is designed to provide a tutorial for
students in computational astrophysics (and hydrodynamics in
general). We introduce simple implementations of some popular methods
used in the field, with the code written to be easily
understandable. All simulations use a single grid (no domain decomposition).

.. note::

   pyro is not meant for demanding scientific simulations—given the
   choice between performance and clarity, clarity is taken.

pyro builds off of a finite-volume framework for solving PDEs. There
are a number of solvers in pyro, allowing for the solution of
hyperbolic (wave), parabolic (diffusion), and elliptic (Poisson)
equations. In particular, the following solvers are developed:

* linear advection

* compressible hydrodynamics

* multigrid

* implicit thermal diffusion

* incompressible hydrodynamics

* low Mach number atmospheric hydrodynamics

Runtime visualization shows the evolution as the equations are solved.

In the pages that follow, the following format is adopted:

* PDF notes provide the basic theory behind the methods.  References
  are cited to provide more detail.

* An overview of the use of the applicable modules from pyro provided.

* Exercises interspersed fill in some of the detail.
