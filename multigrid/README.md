This provides two multigrid solvers for cell-centered data.

## `MG.py`

  This is the basic solver.  It solves constant coefficient
  problems of the form:

  `(alpha - beta L) phi = f`

  where L is the Laplacian.

  The following drivers test it:

  - `mg_test.py`: this solves:

    `u_xx + u_yy = -2[(1-6x**2)y**2(1-y**2) + (1-6y**2)x**2(1-x**2)]`

    at a single resolution on [0,1]x[0,1], with u = 0 on the boundary
    (Dirichlet BCs).

  - `mg_vis.py`: this solves the same problem as `mg_test.py`, but it
    outputs a detailed set of plots at each smoothing iteration showing
    the progression of the solve through the V-cycles


## `variable_coeff_MG.py`

  This is a variable-coefficient solver.  It subclasses `MG.py` and
  extends the basic solver to solve problems with variable
  coefficients:

  `D (eta G phi) = f`

  where D is the divergence and G is the gradient.

  The following drivers test it:

  - `mg_test_vc_constant.py`: this solves the same constant-coefficnet
    Poisson problem as `mg_test.py`, but using the `variable_coeff_MG.py`
    framework.  This makes sure that we can fall back to the simpler
    constant-coefficient case.

  - `mg_test_vc_dirichlet.py`: this solves

    `div . ( alpha grad phi ) = f`

    with

    ```
    alpha = 2.0 + cos(2.0*pi*x)*cos(2.0*pi*y)
    f = -16.0*pi**2*(cos(2*pi*x)*cos(2*pi*y) + 1)*sin(2*pi*x)*sin(2*pi*y)
	```
       
    on [0,1] x [0,1] with homogeneous Dirichlet BCs.  The solution
    is compared to the exact solution and the convergence is measured.

  - `mg_test_vc_periodic.py`: this solves the same problem as
    `mg_test_vc_dirichlet.py`, but with periodic boundary conditions.


## `general_MG.py`

  This is the general multigrid solver, designed to solve elliptic problems
  of the form:

  `alpha phi + div . (beta grad phi) + gamma . grad phi = f`

  The following drivers test it:

  - `mg_test_general_alphabeta_only.py`: this neglects the `gamma` term
     and solves:

     `alpha phi + div . ( beta grad phi ) = f`

     with

     ```
	 alpha = 1.0
     beta = 2.0 + cos(2.0*pi*x)*cos(2.0*pi*y)
     f = (-16.0*pi**2*cos(2*pi*x)*cos(2*pi*y) - 16.0*pi**2 + 1.0)*sin(2*pi*x)*sin(2*pi*y)
	 ```

    on [0,1] x [0,1] with homogeneous Dirichlet BCs.  The solution is
	compared to the exact solution and the convergence is measured.

  - `mg_test_general_beta_only.py`: this neglects both the `alpha` and
    `gamma` terms and solves the same variable-coefficient Poisson problem
	as `mg_test_vc_dirichlet.py` (note that the naming of the coefficients
	differ between those solvers, but the equation solved is the same).
	
  - `mg_test_general_constant.py`: this solves a pure Poisson problem
    (`alpha = gamma = 0; beta = 1`), solving the same problem as
	the base MG solver in `mg_test.py`.

  - `mg_test_general_dirichlet.py`

  - `mg_test_general_inhomogeneous.py`


## `prolong_restrict_demo.py`

  This tests that the restriction and prolongation operations work as
  expected.




