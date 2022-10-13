"""A method-of-lines compressible hydrodynamics solver.  Piecewise
constant reconstruction is done in space and a Runge-Kutta time
integration is used to advance the solutiion.

"""
__all__ = ["simulation"]

from .simulation import Simulation
