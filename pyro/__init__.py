"""pyro is a python hydrodynamics code designed for teaching and prototyping new
methods.
"""

from ._version import version

__version__ = version


from pyro.mesh import (BC, ArrayIndexer, ArrayIndexerFC, CellCenterData2d,
                       FaceCenterData2d, FV2d, Grid2d, RKIntegrator)
from pyro.util import RuntimeParameters, TimerCollection
