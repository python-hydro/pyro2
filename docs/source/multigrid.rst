Multigrid Solvers
=================

pyro solves elliptic problems (like Laplace's equation or Poisson's
equation) through multigrid. This accelerates the convergence of
qsimple relaxation by moving the solution down and up through a series
of grids. Chapter 9 of the `pdf notes <http://open-astrophysics-bookshelf.github.io/numerical_exercises/CompHydroTutorial.pdf>`_ gives an introduction to solving elliptic equations, including multigrid.

There are three solvers:

* The core solver, provided in the class :func:`MG.CellCenterMG2d <pyro.multigrid.MG.CellCenterMG2d>` solves constant-coefficient Helmholtz problems of the form

  $$(\alpha - \beta \nabla^2) \phi = f$$

* The class :func:`variable_coeff_MG.VarCoeffCCMG2d <pyro.multigrid.variable_coeff_MG.VarCoeffCCMG2d>` solves variable coefficient Poisson problems of the form

  $$\nabla \cdot (\eta \nabla \phi ) = f$$

  This class inherits the core functionality from ``MG.CellCenterMG2d``.

* The class :func:`general_MG.GeneralMG2d <pyro.multigrid.general_MG.GeneralMG2d>` solves a general elliptic
  equation of the form

  $$`\alpha \phi + \nabla \cdot ( \beta \nabla \phi) + \gamma \cdot \nabla \phi = f$$

  This class inherits
  the core functionality from :func:`MG.CellCenterMG2d <pyro.multigrid.MG.CellCenterMG2d>`.

  This solver is the only one to support inhomogeneous boundary
  conditions.

We simply use V-cycles in our implementation, and restrict ourselves
to square grids with zoning a power of 2.

.. note::

   The multigrid solver is not controlled through ``pyro_sim.py``
   since there is no time-dependence in pure elliptic problems. Instead,
   there are a few scripts in the multigrid/ subdirectory that
   demonstrate its use.
