# unit tests

import numpy as np
from numpy.testing import assert_array_equal

import pyro.mesh.patch as patch
import pyro.multigrid.edge_coeffs as edge_coeffs
import pyro.multigrid.MG as MG


# utilities
def test_edge_coeffs():
    # make dx = dy = 1 so the normalization is trivial
    g = patch.Grid2d(4, 6, ng=2, xmax=4, ymax=6)

    # we'll fill ghost cells ourself
    b = np.arange(g.qx, dtype=np.float64)

    eta1 = g.scratch_array()
    eta1[:, :] = b[:, np.newaxis]

    e1 = edge_coeffs.EdgeCoeffs(g, eta1)

    assert_array_equal(e1.x[:, g.jc],
                       np.array([0., 0., 1.5, 2.5, 3.5, 4.5, 5.5, 0.]))

    b = np.arange(g.qy, dtype=np.float64)

    eta2 = g.scratch_array()
    eta2[:, :] = b[np.newaxis, :]

    e2 = edge_coeffs.EdgeCoeffs(g, eta2)

    assert_array_equal(e2.y[g.ic, :],
                       np.array([0., 0., 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 0.]))


# test the gradient stuff -- we don't actually need to do a solve, just
# initialize a phi and get the gradient
def test_mg_gradient():
    a = MG.CellCenterMG2d(8, 8, ng=1, xmax=8, ymax=8)

    s = a.soln_grid.scratch_array()

    s.v()[:, :] = np.fromfunction(lambda i, j: i*(s.g.nx-i-1)*j*(s.g.ny-j-1),
                                 (s.g.nx, s.g.ny))

    a.init_solution(s)

    a.grids[a.nlevels-1].fill_BC("v")

    gx, gy = a.get_solution_gradient()

    assert_array_equal(gx[:, gx.g.jc],
                       np.array([0., 36., 60., 36., 12., -12., -36., -60., -36., 0.]))

    assert_array_equal(gy[gx.g.ic, :],
                       np.array([0., 36., 60., 36., 12., -12., -36., -60., -36., 0.]))
