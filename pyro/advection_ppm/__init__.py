"""
The Piecewise Parabolic reconstruction Method (PPM) + CTU unsplit dimensional implementation. This problem is a second-order
advection solver, based on the Colella & Woodward (1984), Colella (1990), Colella & Miller (2002) and Colella & Sekora (2002)
papers descriptions. The advantage of choosing the PPM parabolic reconstruction instead of the Piecewise Linear Reconstruction
(PLR), is the increased resolution of interfaces at contact discontinuities.

"""

__all__ = ['simulation']
from .simulation import Simulation
