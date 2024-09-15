*****************************
Low Mach number hydrodynamics
*****************************

pyro's low Mach hydrodynamics solver is designed for atmospheric
flows. It captures the effects of stratification on a fluid element by
enforcing a divergence constraint on the velocity field.  The
governing equations are:

.. math::

   \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho U) &= 0 \\
   \frac{\partial U}{\partial t} + U \cdot \nabla U + \frac{\beta_0}{\rho} \nabla \left ( \frac{p'}{\beta_0} \right ) &= \frac{\rho'}{\rho} g \\
   \nabla \cdot (\beta_0 U) = 0

with :math:`\nabla p_0 = \rho_0 g` and :math:`\beta_0 = p_0^{1/\gamma}`.

``lm_atm`` solver
=================

As with the incompressible solver, we implement a cell-centered approximate projection method.

The main parameters that affect this solver are:

.. include:: lm_atm_defaults.inc

.. include:: lm_atm_problems.inc
