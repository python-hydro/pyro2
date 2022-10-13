r"""This is the pyro multigrid solver.  There are several versions.

MG implements a second-order discretization of a constant-coefficient
Helmholtz equation:

.. math::

   (\alpha - \beta L) \phi = f

variable_coeff_MG implements a variable-coefficient Poisson equation

.. math::

   \nabla \cdot { \eta \nabla \phi } = f

general_MG implements a more general elliptic equation

.. math::

   \alpha \phi + \nabla \cdot { \beta \nabla \phi } + \gamma \cdot \nabla \phi = f


All use pure V-cycles to solve elliptic problems

"""

__all__ = ['MG', 'variable_coeff_MG', 'general_MG', 'edge_coeffs']
