"""
The pyro compressible hydrodynamics solver with reactions.  This
implements the second-order (piecewise-linear), unsplit method of
Colella 1990, and incorporates reactions via Strang splitting.

"""
__all__ = ["simulation"]

from .simulation import Simulation
