#!/usr/bin/env python3

"""

an example of using the multigrid class to solve Laplace's equation.  Here, we
solve::

   u_xx + u_yy = -2[(1-6x**2)y**2(1-y**2) + (1-6y**2)x**2(1-x**2)]
   u = 0 on the boundary

this is the example from page 64 of the book `A Multigrid Tutorial, 2nd Ed.`

The analytic solution is u(x,y) = (x**2 - x**4)(y**4 - y**2)

"""


import matplotlib.pyplot as plt
import numpy as np

import pyro.multigrid.MG as MG


# the analytic solution
def true(x, y):
    return (x**2 - x**4)*(y**4 - y**2)


# the righthand side
def f(x, y):
    return -2.0*((1.0-6.0*x**2)*y**2*(1.0-y**2) + (1.0-6.0*y**2)*x**2*(1.0-x**2))


def doit(nx, ny):
    # test the multigrid solver

    # create the multigrid object
    a = MG.CellCenterMG2d(nx, ny,
                          xl_BC_type="dirichlet", yl_BC_type="dirichlet",
                          xr_BC_type="dirichlet", yr_BC_type="dirichlet",
                          verbose=0,
                          nsmooth=5, nsmooth_bottom=10,
                          vis=1, true_function=true,
                          vis_title=r"$u_{xx} + u_{yy} = -2[(1-6x^2)y^2(1-y^2) + (1-6y^2)x^2(1-x^2)]$")

    plt.ion()

    plt.figure(num=1, figsize=(12.8, 7.2), dpi=100, facecolor='w')

    # initialize the solution to 0
    init = a.soln_grid.scratch_array()

    a.init_solution(init)

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
          (a.soln_grid.norm(e), a.relative_error, a.num_cycles))

    # plot it
    # plt.figure(num=1, figsize=(2.10,2.10), dpi=100, facecolor='w')
    plt.figure(num=1, figsize=(5.0, 5.0), dpi=100, facecolor='w')

    plt.imshow(np.transpose(v[a.ilo:a.ihi+1, a.jlo:a.jhi+1]),
               interpolation="nearest", origin="lower",
               extent=[a.xmin, a.xmax, a.ymin, a.ymax])

    # plt.axis("off")
    # plt.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=1.0)

    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig("mg_test.png")

    # store the output for later comparison
    my_data = a.get_solution_object()
    my_data.write("mg_test")


if __name__ == "__main__":
    doit(64, 64)
