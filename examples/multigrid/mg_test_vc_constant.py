#!/usr/bin/env python3

"""

Test the variable coefficient MG solver with a CONSTANT coefficient
problem -- the same one from the multigrid class test.  This ensures
we didn't screw up the base functionality here.

We solve::

   u_xx + u_yy = -2[(1-6x**2)y**2(1-y**2) + (1-6y**2)x**2(1-x**2)]
   u = 0 on the boundary

this is the example from page 64 of the book `A Multigrid Tutorial, 2nd Ed.`

The analytic solution is u(x,y) = (x**2 - x**4)(y**4 - y**2)

"""


import matplotlib.pyplot as plt
import numpy as np

import pyro.mesh.boundary as bnd
import pyro.mesh.patch as patch
import pyro.multigrid.variable_coeff_MG as MG


# the analytic solution
def true(x, y):
    return (x**2 - x**4)*(y**4 - y**2)


# the coefficients
def alpha(x, y):
    return np.ones_like(x)


# the righthand side
def f(x, y):
    return -2.0*((1.0-6.0*x**2)*y**2*(1.0-y**2) + (1.0-6.0*y**2)*x**2*(1.0-x**2))


def test_vc_constant(N):

    # test the multigrid solver
    nx = N
    ny = nx

    # create the coefficient variable -- note we don't want Dirichlet here,
    # because that will try to make alpha = 0 on the interface.  alpha can
    # have different BCs than phi
    g = patch.Grid2d(nx, ny, ng=1)
    d = patch.CellCenterData2d(g)
    bc_c = bnd.BC(xlb="neumann", xrb="neumann",
                  ylb="neumann", yrb="neumann")
    d.register_var("c", bc_c)
    d.create()

    c = d.get_var("c")
    c[:, :] = alpha(g.x2d, g.y2d)

    plt.clf()

    plt.figure(num=1, figsize=(5.0, 5.0), dpi=100, facecolor='w')

    img1 = plt.imshow(np.transpose(c[g.ilo:g.ihi+1, g.jlo:g.jhi+1]),
               interpolation="nearest", origin="lower",
               extent=[g.xmin, g.xmax, g.ymin, g.ymax])

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"nx = {nx}")

    plt.colorbar(img1)

    plt.savefig("mg_alpha.png")

    # check whether the RHS sums to zero (necessary for periodic data)
    rhs = f(g.x2d, g.y2d)
    print(f"rhs sum: {np.sum(rhs[g.ilo:g.ihi+1, g.jlo:g.jhi+1])}")

    # create the multigrid object
    a = MG.VarCoeffCCMG2d(nx, ny,
                          xl_BC_type="dirichlet", yl_BC_type="dirichlet",
                          xr_BC_type="dirichlet", yr_BC_type="dirichlet",
                          coeffs=c, coeffs_bc=bc_c,
                          verbose=1)

    # initialize the solution to 0
    a.init_zeros()

    # initialize the RHS using the function f
    rhs = f(a.x2d, a.y2d)
    a.init_RHS(rhs)

    # solve to a relative tolerance of 1.e-11
    a.solve(rtol=1.e-11)

    # alternately, we can just use smoothing by uncommenting the following
    # a.smooth(a.nlevels-1,50000)

    # get the solution
    v = a.get_solution()

    # compute the error from the analytic solution
    b = true(a.x2d, a.y2d)
    e = v - b

    print(" L2 error from true solution = %g\n rel. err from previous cycle = %g\n num. cycles = %d" %
          (e.norm(), a.relative_error, a.num_cycles))

    # plot it
    plt.clf()

    plt.figure(num=1, figsize=(10.0, 5.0), dpi=100, facecolor='w')

    plt.subplot(121)

    img2 = plt.imshow(np.transpose(v[a.ilo:a.ihi+1, a.jlo:a.jhi+1]),
               interpolation="nearest", origin="lower",
               extent=[a.xmin, a.xmax, a.ymin, a.ymax])

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"nx = {nx}")

    plt.colorbar(img2)

    plt.subplot(122)

    img3 = plt.imshow(np.transpose(e[a.ilo:a.ihi+1, a.jlo:a.jhi+1]),
               interpolation="nearest", origin="lower",
               extent=[a.xmin, a.xmax, a.ymin, a.ymax])

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("error")

    plt.colorbar(img3)

    plt.tight_layout()

    plt.savefig("mg_test.png")

    # store the output for later comparison
    my_data = a.get_solution_object()
    my_data.write("mg_test")


if __name__ == "__main__":
    test_vc_constant(256)
