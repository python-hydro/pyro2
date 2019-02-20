"""This is a 4th order accurate compressible hydrodynamics solver,
implementing the spatial reconstruction from McCorquodale & Colella
(2011) but using an SDC scheme for the time integration.

"""
__all__ = ["simulation"]

from .simulation import Simulation
