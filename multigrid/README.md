This provides two multigrid solvers for cell-centered data.

## `MG.py`

  This is the basic solver.  It solves constant coefficient
  problems of the form:

  `(alpha - beta L) phi = f`

  where L is the Laplacian.

  The following drivers test it:

  - `test_mg.py`: this solves:

    `u_xx + u_yy = -2[(1-6x**2)y**2(1-y**2) + (1-6y**2)x**2(1-x**2)]`

    at a single resolution on [0,1]x[0,1], with u = 0 on the boundary
    (Dirichlet BCs).

  - `mg_vis.py`: this solves the same problem as `test_mg.py`, but it
    outputs a detailed set of plots at each smoothing iteration showing
    the progression of the solve through the V-cycles


## `variable_coeff_MG.py`

  This is a variable-coefficient solver.  It subclasses `MG.py` and
  extends the basic solver to solve problems with variable
  coefficients:

  D (eta G phi) = f

  where D is the divergence and G is the gradient.

  The following drivers test it:

  - `test_mg_vc_constant.py`: this solves the same constant-coefficnet
    Poisson problem as `test_mg.py`, but using the `variable_coeff_MG.py`
    framework.  This makes sure that we can fall back to the simpler
    constant-coefficient case.

  - `test_mg_vc_dirichlet.py`: this solves

    div . ( alpha grad phi ) = f

    with

    `alpha = 2.0 + cos(2.0*pi*x)*cos(2.0*pi*y)`

    `f = -16.0*pi**2*(cos(2*pi*x)*cos(2*pi*y) + 1)*sin(2*pi*x)*sin(2*pi*y)`
       
    on [0,1] x [0,1] with homogeneous Dirichlet BCs.  The solution
    is compared to the exact solution and the convergence is measured.

  - `test_mg_vc_periodic.py`: this solves the same problem as
    `test_mg_vc_dirichlet.py`, but with periodic boundary conditions.



## `prolong_restrict_test.py`

  This tests that the restriction and prolongation operations work as
  expected.



