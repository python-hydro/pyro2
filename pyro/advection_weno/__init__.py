"""The pyro advection solver.  This implements a finite difference
Lax-Friedrichs flux split method with WENO reconstruction based on
Shu's review from 1998
(https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980007543.pdf)
although the notation more follows Gerolymos et al
(https://doi.org/10.1016/j.jcp.2009.07.039).

Most of the code is taken from advection_rk and toy-conslaw.

The general flow of the solver when invoked through pyro.py is:

-  create grid

-  initial conditions

-  main loop

     * fill ghost cells

     * compute dt

     * compute fluxes

     * conservative update

     * output

"""

__all__ = ['simulation']
from .simulation import Simulation
