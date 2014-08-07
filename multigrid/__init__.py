"""
This is the pyro multigrid solver.  It uses a second-order discretization
of a constant-coefficient Helmholtz equation:

 (alpha - beta L) phi = f

and pure V-cycles to solve elliptic problems
"""

__all__ = ['MG', 'variable_coeff_MG']

