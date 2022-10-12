"""pyro is a python hydrodynamics code designed for teaching and prototyping new
methods.
"""

from ._version import version

__version__ = version


from pyro2.mesh import (ArrayIndexer, ArrayIndexerFC, BC, FV2d,
                        RKIntegrator, Grid2d,
                        CellCenterData2d, FaceCenterData2d)

from pyro2.util import TimerCollection, RuntimeParameters



