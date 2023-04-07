"""
The pyro inviscid burgers solver.  This implements a second-order,
unsplit method for inviscid burgers equations based on the Colella 1990 paper.
"""

__all__ = ['simulation']
from .simulation import Simulation
