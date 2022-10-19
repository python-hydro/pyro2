#!/usr/bin/env python3

"""

an example of using the multigrid class to solve Laplace's equation.  Here, we
solve::

   u_xx + u_yy = -2[(1-6x**2)y**2(1-y**2) + (1-6y**2)x**2(1-x**2)]
   u = 0 on the boundary

this is the example from page 64 of the book `A Multigrid Tutorial, 2nd Ed.`

The analytic solution is u(x,y) = (x**2 - x**4)(y**4 - y**2)

"""


import os

import matplotlib.pyplot as plt
import numpy as np

from pyro.util import compare
import pyro.multigrid.MG as MG
import pyro.util.io_pyro as io
import pyro.util.msg as msg


# the analytic solution
def true(x, y):
    return (x**2 - x**4)*(y**4 - y**2)


# the righthand side
def f(x, y):
    return -2.0*((1.0-6.0*x**2)*y**2*(1.0-y**2) + (1.0-6.0*y**2)*x**2*(1.0-x**2))


def test_poisson_dirichlet(N, store_bench=False, comp_bench=False, bench_dir="tests/",
                           make_plot=False, verbose=1, rtol=1e-12):

    # test the multigrid solver
    nx = N
    ny = nx

    # create the multigrid object
    a = MG.CellCenterMG2d(nx, ny,
                          xl_BC_type="dirichlet", yl_BC_type="dirichlet",
                          xr_BC_type="dirichlet", yr_BC_type="dirichlet",
                          verbose=verbose)

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
    if make_plot:
        plt.figure(num=1, figsize=(5.0, 5.0), dpi=100, facecolor='w')

        plt.imshow(np.transpose(v[a.ilo:a.ihi+1, a.jlo:a.jhi+1]),
                   interpolation="nearest", origin="lower",
                   extent=[a.xmin, a.xmax, a.ymin, a.ymax])

        plt.xlabel("x")
        plt.ylabel("y")

        print("Saving figure to mg_test.png")

        plt.savefig("mg_test.png")

    # store the output for later comparison
    bench = "mg_poisson_dirichlet"

    my_data = a.get_solution_object()

    if store_bench:
        my_data.write(f"{bench_dir}/{bench}")

    # do we do a comparison?
    if comp_bench:
        compare_file = f"{bench_dir}/{bench}"
        msg.warning("comparing to: %s " % (compare_file))
        bench_data = io.read(compare_file)

        result = compare.compare(my_data, bench_data, rtol)

        if result == 0:
            msg.success(f"results match benchmark to within relative tolerance of {rtol}\n")
        else:
            msg.warning("ERROR: " + compare.errors[result] + "\n")

        return result

    return None


if __name__ == "__main__":
    test_poisson_dirichlet(256, comp_bench=True, make_plot=True)
