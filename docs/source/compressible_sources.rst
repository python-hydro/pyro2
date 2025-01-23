*************************
Compressible source terms
*************************

Implementation
==============

Various source terms are included in the compressible equation set.
Their implementation is solver-dependent.

``compressible``
----------------

For the ``compressible`` solver, source terms are included in the
compressible equations via a predictor-corrector formulation.
Starting with the conserved state, $\Uc^{n}$, we do the update as:

.. math::

   \begin{align*}
   \Uc^{n+1,\star} &= \Uc^n - \Delta t \left [ \nabla \cdot {\bf F}(\Uc)\right ]^{n+1/2} + \Delta t {\bf S}(\Uc^n) \\
   \Uc^{n+1} &= \Uc^{n+1,\star} + \frac{\Delta t}{2} ({\bf S}(\Uc^{n+1,\star}) - {\bf S}(\Uc^n))
   \end{align*}

This time-centers the sources, making the update second-order accurate.


``compressible_rk``, ``compressible_fv4``, ``compressible_sdc``
---------------------------------------------------------------

These solvers all use a method-of-lines type discretization, so at each temporal node / stage, $s$,
that the update is needed, we compute it as:

.. math::

   \left . \frac{\partial \Uc}{\partial t} \right |^s = -\nabla \cdot {\bf F}(\Uc^s) + {\bf S}(\Uc^s)

and the method's time-integration strategy ensures that it achieves
the correct order of accuracy.

Gravity
=======

All solvers include the ability to adds a constant gravitational source,
set by the parameter ``compressible.grav``.

For the ``compressible`` solver, the corrector update is done by first
updating the momentum, and then using the new momentum to get the
final total energy.  This makes it appear as if it were an
implicit-in-time update.

Sponge
======

A sponge is a damping term in the velocity meant to drive the velocity
to zero in regions that are not interesting.  This is often used with
outflow / zero-gradient boundary conditions in atmospheric calculations
for the region above the atmosphere (see the ``convection`` problem
for an example).

The sponge is enabled by setting ``sponge.do_sponge=1``.

The momentum equation in this case appears as:

.. math::

   \frac{\partial (\rho \Ub)}{\partial t} + \nabla \cdot (\rho \Ub \Ub) + \nabla p = \rho {\bf g} - \frac{f_\mathrm{damp}}{\tau_\mathrm{damp}} \rho \Ub

where

* $f_\mathrm{damp}$ is in $[0, 1]$ and indicates where the
  sponge is active.  This is specified in terms of density
  and takes the form:

  .. math::

     f_\mathrm{damp} = \left \{ \begin{array}{cc} 0 & \rho > \rho_\mathrm{begin} \\
                                                  \frac{1}{2} \left ( 1 - \cos\left(\pi \frac{\rho - \rho_\mathrm{begin}}{\rho_\mathrm{full} - \rho_\mathrm{begin}} \right ) \right ) & \rho_\mathrm{begin} > \rho > \rho_\mathrm{full} \\
                                                  1 & \rho < \rho_\mathrm{full} \end{array} \right .

  where $\rho_\mathrm{begin}$ is the density where the sponge turns on
  (set by ``sponge.sponge_rho_begin``) and $\rho_\mathrm{full}$ is the
  density where the sponge is fully enabled (set by
  ``sponge.sponge_rho_begin``), with $\rho_\mathrm{begin} >
  \rho_\mathrm{full}$.

* $\tau_\mathrm{damp}$ is the timescale for the sponge to
  act (typically this should be a few to 10 timesteps).
  This is set by ``sponge.sponge_timescale``.




Problem-dependent source
========================

A problem-dependent source can be added to the conserved equations.
This is provided by a function ``source_terms`` in the problem's
module, with the form:

.. code:: python

   source_terms(myg, U, ivars, rp)

This is evaluated using the appropriate conserved state, $\Uc$, and
expected to return a vector ${\bf S}(\Uc)$.  See the ``convection``
problem for an example (as a heating term).
