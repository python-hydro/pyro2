"""This is the pyro multigrid solver.  THere are several versions.

MG implements a second-order discretization of a constant-coefficient
Helmholtz equation:

 (alpha - beta L) phi = f


variable_coeff_MG implements a variable-coefficient Poisson equation:

  div { eta grad phi } = f

general_MG implements a more general elliptic equation:

  alpha phi + div { beta grad phi } + gamma . grad phi = f


All use pure V-cycles to solve elliptic problems

"""

__all__ = ['MG', 'variable_coeff_MG', 'general_MG', 'edge_coeffs.py']
