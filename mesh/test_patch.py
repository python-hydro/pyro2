# unit tests for the patch
import mesh.patch as patch
import numpy as np

from numpy.testing import assert_array_equal
from nose.tools import assert_equal

def test_dx_dy():
    g = patch.Grid2d(4, 6, ng=2, ymax=1.5)

    assert_equal(g.dx, 0.25)
    assert_equal(g.dy, 0.25)

def test_grid_coords():
    g = patch.Grid2d(4, 6, ng=2, ymax=1.5)

    assert_array_equal(g.x[g.ilo:g.ihi+1], np.array([0.125, 0.375, 0.625, 0.875]))
    assert_array_equal(g.y[g.jlo:g.jhi+1], np.array([0.125, 0.375, 0.625, 0.875, 1.125, 1.375]))

def test_grid_2d_coords():
    g = patch.Grid2d(4, 6, ng=2, ymax=1.5)

    assert_array_equal(g.x, g.x2d[:,g.jc])
    assert_array_equal(g.y, g.y2d[g.ic,:])

def test_course_like():
    g = patch.Grid2d(4, 6, ng=2, ymax=1.5)
    c = g.coarse_like(2)

    assert_equal(c.nx, 2)
    assert_equal(c.ny, 3)


