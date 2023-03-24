Burgers' Equations
==================

Burgers' Equations are nonlinear hyperbolic equations. It has the same form as the advection equation, except that the quantity that we're advecting is the velocity itself.

``Inviscid Burgers``
--------------------------------

A 2D Burgers' Equation has the following form:

.. math::

   u_t + u u_x + v u_y = 0\\
   v_t + u v_x + v v_y = 0

Here we have two 2D advection equations, where the x-velocity, :math:`u`, and y-velocity, :math:`v`, are the two quantities that we wish to advect with.

:py:mod:`pyro.burgers` is modified based on the :py:mod:`pyro.advection` with a different Riemann solver and timestep restriction.

Since velocity is no longer a constant, the timestep is now restricted to the each minimum velocity in each cell:

.. math::

   \Delta t < \min \left \{ \min \left \{ \frac{\Delta x}{|u_i|} \right \}, \min \left \{ \frac{\Delta y}{|v_i|}  \right \}  \right \}

The main difference of Burgers equation compared to the linear advection equation is the creation of shock and rarefactions due velocity been non-constant. This introduces a slightly different Riemann's problem which depends on shock speed by using the *Rankine-Hugoniot* jump condition.

The parameters for this solver are:

.. include:: burgers_defaults.inc
