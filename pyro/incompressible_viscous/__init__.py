"""
The pyro solver for incompressible and viscous flow.  This implements a
second-order approximate projection method.  The general flow is:

* create the limited slopes of u and v (in both directions)

* get the advective velocities through a piecewise linear Godunov
  method

* enforce the divergence constraint on the velocities through a
  projection (the MAC projection)

* recompute the interface states using the new advective velocity

* Solve for U in a (decoupled) parabolic solve including viscosity term

* project the final velocity to enforce the divergence constraint.

The projections are done using multigrid
"""

__all__ = ["simulation"]

from .simulation import Simulation
