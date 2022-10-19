#!/usr/bin/env python3

"""Test the general MG solver with inhomogeneous Dirichlet
   boundary conditions.

Here we solve::

   alpha phi + div{beta grad phi} + gamma . grad phi = f

with::

   alpha = 10.0
   beta = x*y + 1  (note: x*y alone doesn't work)
   gamma_x = 1
   gamma_y = 1

   f =  -(pi/2)*(x + 1)*sin(pi*y/2)*cos(pi*x/2)
       - (pi/2)*(y + 1)*sin(pi*x/2)*cos(pi*y/2) +
       (-pi**2*(x*y+1)/2 + 10)*cos(pi*x/2)*cos(pi*y/2)

This has the exact solution::

   phi = cos(pi*x/2)*cos(pi*y/2)

on [0,1] x [0,1], with Dirichlet boundary conditions::

   phi(x=0) = cos(pi*y/2)
   phi(x=1) = 0
   phi(y=0) = cos(pi*x/2)
   phi(y=1) = 0

For the coefficients we do not have to impose the same BCs, since that
may represent a different physical quantity.  beta is the one that
really matters since it must be brought to the edges.  Here we take
beta to have Neumann BCs.  (Dirichlet BCs for beta will force it to 0
on the boundary, which is not correct here)

"""


import os

import matplotlib.pyplot as plt
import numpy as np

from pyro.util import compare
import pyro.mesh.boundary as bnd
import pyro.mesh.patch as patch
import pyro.multigrid.general_MG as MG
import pyro.util.io_pyro as io
import pyro.util.msg as msg


# the analytic solution
def true(x, y):
    return np.cos(np.pi*x/2.0)*np.cos(np.pi*y/2.0)


# the coefficients
def alpha(x, y):
    return 10.0*np.ones_like(x)


def beta(x, y):
    return x*y + 1.0


def gamma_x(x, y):
    return np.ones_like(x)


def gamma_y(x, y):
    return np.ones_like(x)


# the righthand side
def f(x, y):
    return -0.5*np.pi*(x + 1.0)*np.sin(np.pi*y/2.0)*np.cos(np.pi*x/2.0) - \
        0.5*np.pi*(y + 1.0)*np.sin(np.pi*x/2.0)*np.cos(np.pi*y/2.0) + \
        (-np.pi**2*(x*y+1.0)/2.0 + 10.0)*np.cos(np.pi*x/2.0)*np.cos(np.pi*y/2.0)


# boundary conditions
def xl_func(y):
    return np.cos(np.pi*y/2.0)


def yl_func(x):
    return np.cos(np.pi*x/2.0)


def test_general_poisson_inhomogeneous(N, store_bench=False, comp_bench=False, bench_dir="tests/",
                                       make_plot=False, verbose=1, rtol=1.e-12):
    """
    test the general MG solver.  The return value
    here is the error compared to the exact solution, UNLESS
    comp_bench=True, in which case the return value is the
    error compared to the stored benchmark
    """

    # test the multigrid solver
    nx = N
    ny = nx

    # create the coefficient variable
    g = patch.Grid2d(nx, ny, ng=1)
    d = patch.CellCenterData2d(g)
    bc_c = bnd.BC(xlb="neumann", xrb="neumann",
                  ylb="neumann", yrb="neumann")
    d.register_var("alpha", bc_c)
    d.register_var("beta", bc_c)
    d.register_var("gamma_x", bc_c)
    d.register_var("gamma_y", bc_c)
    d.create()

    a = d.get_var("alpha")
    a[:, :] = alpha(g.x2d, g.y2d)

    b = d.get_var("beta")
    b[:, :] = beta(g.x2d, g.y2d)

    gx = d.get_var("gamma_x")
    gx[:, :] = gamma_x(g.x2d, g.y2d)

    gy = d.get_var("gamma_y")
    gy[:, :] = gamma_y(g.x2d, g.y2d)

    # create the multigrid object
    a = MG.GeneralMG2d(nx, ny,
                       xl_BC_type="dirichlet", yl_BC_type="dirichlet",
                       xr_BC_type="dirichlet", yr_BC_type="dirichlet",
                       xl_BC=xl_func,
                       yl_BC=yl_func,
                       coeffs=d,
                       verbose=verbose, vis=0, true_function=true)

    # initialize the solution to 0
    a.init_zeros()

    # initialize the RHS using the function f
    rhs = f(a.x2d, a.y2d)
    print(np.min(rhs), np.max(rhs))

    a.init_RHS(rhs)

    # solve to a relative tolerance of 1.e-10
    a.solve(rtol=1.e-10)

    # alternately, we can just use smoothing by uncommenting the following
    # a.smooth(a.nlevels-1,50000)

    # get the solution
    v = a.get_solution()

    # compute the error from the analytic solution
    b = true(a.x2d, a.y2d)
    e = v - b

    enorm = e.norm()
    print(" L2 error from true solution = %g\n rel. err from previous cycle = %g\n num. cycles = %d" %
          (enorm, a.relative_error, a.num_cycles))

    # plot the solution
    if make_plot:
        plt.clf()

        plt.figure(figsize=(10.0, 4.0), dpi=100, facecolor='w')

        plt.subplot(121)

        img1 = plt.imshow(np.transpose(v.v()),
                   interpolation="nearest", origin="lower",
                   extent=[a.xmin, a.xmax, a.ymin, a.ymax])

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"nx = {nx}")

        plt.colorbar(img1)

        plt.subplot(122)

        img2 = plt.imshow(np.transpose(e.v()),
                   interpolation="nearest", origin="lower",
                   extent=[a.xmin, a.xmax, a.ymin, a.ymax])

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("error")

        plt.colorbar(img2)

        plt.tight_layout()

        plt.savefig("mg_general_inhomogeneous_test.png")

    # store the output for later comparison
    bench = "mg_general_poisson_inhomogeneous"

    my_data = a.get_solution_object()

    if store_bench:
        my_data.write(f"{bench_dir}/{bench}")

    # do we do a comparison?
    if comp_bench:
        compare_file = f"{bench_dir}/{bench}"
        msg.warning("comparing to: %s " % (compare_file))
        bench = io.read(compare_file)

        result = compare.compare(my_data, bench, rtol)

        if result == 0:
            msg.success(f"results match benchmark to within relative tolerance of {rtol}\n")
        else:
            msg.warning("ERROR: " + compare.errors[result] + "\n")

        return result

    # normal return -- error wrt true solution
    return enorm


if __name__ == "__main__":

    N = [16, 32, 64, 128, 256, 512]
    err = []

    plot = False
    store = False
    do_compare = False

    for nx in N:
        if nx == max(N):
            plot = True

        enorm = test_general_poisson_inhomogeneous(nx, make_plot=plot,
                                                   store_bench=store, comp_bench=do_compare)

        err.append(enorm)

    # plot the convergence
    N = np.array(N, dtype=np.float64)
    err = np.array(err)

    plt.clf()
    plt.loglog(N, err, "x", color="r")
    plt.loglog(N, err[0]*(N[0]/N)**2, "--", color="k")

    plt.xlabel("N")
    plt.ylabel("error")

    plt.ylim(1.e-7, 1.e-2)

    fig = plt.gcf()
    fig.set_size_inches(7.0, 6.0)

    plt.tight_layout()

    plt.savefig("mg_general_inhomogeneous_converge.png")
