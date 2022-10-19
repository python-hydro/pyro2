import numpy as np
from numpy.testing import assert_array_equal

import pyro.mesh.array_indexer as ai
import pyro.mesh.patch as patch


# utilities
def test_buf_split():
    assert_array_equal(ai._buf_split(2), [2, 2, 2, 2])
    assert_array_equal(ai._buf_split((2, 3)), [2, 3, 2, 3])


# ArrayIndexer tests
def test_indexer():
    g = patch.Grid2d(2, 3, ng=2)
    a = g.scratch_array()

    a[:, :] = np.arange(g.qx*g.qy).reshape(g.qx, g.qy)

    assert_array_equal(a.v(), np.array([[16., 17., 18.], [23., 24., 25.]]))

    assert_array_equal(a.ip(1), np.array([[23., 24., 25.], [30., 31., 32.]]))
    assert_array_equal(a.ip(-1), np.array([[9., 10., 11.], [16., 17., 18.]]))

    assert_array_equal(a.jp(1), np.array([[17., 18., 19.], [24., 25., 26.]]))
    assert_array_equal(a.jp(-1), np.array([[15., 16., 17.], [22., 23., 24.]]))

    assert_array_equal(a.ip_jp(1, 1), np.array([[24., 25., 26.], [31., 32., 33.]]))


def test_is_symmetric():
    g = patch.Grid2d(4, 3, ng=0)
    a = g.scratch_array()

    a[:, 0] = [1, 2, 2, 1]
    a[:, 1] = [2, 4, 4, 2]
    a[:, 2] = [1, 2, 2, 1]

    assert a.is_symmetric()


def test_is_asymmetric():
    g = patch.Grid2d(4, 3, ng=0)
    a = g.scratch_array()

    a[:, 0] = [-1, -2, 2, 1]
    a[:, 1] = [-2, -4, 4, 2]
    a[:, 2] = [-1, -2, 2, 1]

    assert a.is_asymmetric()
