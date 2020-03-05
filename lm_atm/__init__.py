"""The pyro solver for low Mach number atmospheric flow.  This
implements as second-order approximate projection method.  The general
flow is:

* create the limited slopes of rho, u and v (in both directions)

* get the advective velocities through a piecewise linear Godunov
  method

* enforce the divergence constraint on the velocities through a
  projection (the MAC projection)

* predict rho to edges and do the conservative update

* recompute the interface states using the new advective velocity

* update U in time to get the provisional velocity field

* project the final velocity to enforce the divergence constraint.

The projections are done using multigrid

"""

__all__ = ["simulation"]

from .simulation import Basestate, Simulation
