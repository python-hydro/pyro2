**********************
Compressible exercises
**********************

Explorations
============

* Measure the growth rate of the Rayleigh-Taylor instability for
  different wavenumbers.

* There are multiple Riemann solvers in the compressible
  algorithm. Run the same problem with the different Riemann solvers
  and look at the differences. Toro's text is a good book to help
  understand what is happening.

* Run the problems with and without limiting—do you notice any overshoots?


Extensions
==========

* Limit on the characteristic variables instead of the primitive
  variables. What changes do you see? (the notes show how to implement
  this change.)

* Add passively advected species to the solver.

* Add an external heating term to the equations.

* Add 2-d axisymmetric coordinates (r-z) to the solver. This is
  discussed in the notes. Run the Sedov problem with the explosion on
  the symmetric axis—now the solution will behave like the spherical
  sedov explosion instead of the cylindrical explosion.

* Swap the piecewise linear reconstruction for piecewise parabolic
  (PPM). The notes and the Miller and Colella paper provide a good basis
  for this.  Research the Roe Riemann solver and implement it in pyro.
