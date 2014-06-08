"""
The pyro advection solver.  This implements a second-order, unsplit
method for linear advection based on the Colella 1990 paper.

The general flow of the solver when invoked through pyro.py is:

  create grid

  initial conditions

  main loop

     fill ghost cells

     compute dt

     compute fluxes

     conservative update

     output

"""

__all__ = ['initialize','evolve','preevolve']
from initialize import *
from preevolve import *
from evolve import *
from timestep import *
from dovis import *
