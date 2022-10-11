"""The pyro diffusion solver.  This implements second-order implicit
diffusion using Crank-Nicolson time-differencing.  The resulting
system is solved using multigrid.

The general flow is:

* compute the RHS given the current state

* set up the MG

* solve the system using MG for updated phi

The timestep is computed as::

   CFL* 0.5*dt/dx**2

"""

__all__ = ["simulation"]

from .simulation import Simulation
