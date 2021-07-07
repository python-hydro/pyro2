# unit tests for mapped
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import sympy
from sympy.abc import x, y

import mesh.mapped as mapped


class TestMappedGrid2d(object):

    @classmethod
    def setup_class(cls):
        """ this is run once for each class before any tests """
        pass

    @classmethod
    def teardown_class(cls):
        """ this is run once for each class after all tests """
        pass

    def setup_method(self):
        """ this is run before each test """
        pass

    def teardown_method(self):
        """ this is run after each test """
        pass

    def test_rectilinear(self):
        """
        Test mapped grid class for a rectilinear grid
        """

        def map(myg):
            return sympy.Matrix([2 * x, 0.1 * y])

        grid = mapped.MappedGrid2d(map, 4, 8, ng=2, ymax=2.0)

        assert grid.dx == 0.25
        assert grid.dy == 0.25

        assert_array_almost_equal(
            grid.kappa, np.ones((8, 12)) * 0.2, decimal=12)
        assert_array_almost_equal(
            grid.gamma_fcx, np.ones((8, 12)) * 0.1, decimal=12)
        assert_array_almost_equal(
            grid.gamma_fcy, np.ones((8, 12)) * 2, decimal=12)

        assert_array_almost_equal(grid.R_fcx(
            2, 0, 1)[0, 0, :, :], np.eye(2), decimal=12)
        assert_array_almost_equal(grid.R_fcy(
            2, 0, 1)[-1, -1, :, :], np.eye(2), decimal=12)

    def test_noncartesian(self):
        """
        Test mapped grid class for a non cartesian grid.
        """

        def map(myg):
            return sympy.Matrix([0.5 * x + y, x - 0.4 * y])

        grid = mapped.MappedGrid2d(map, 4, 4, ng=2)

        # numpy and scipy use slightly different sqrt routines so lose a little
        # bit of precision - use almost equal here rather than equal
        assert_array_almost_equal(grid.kappa, np.ones((8, 8)) * 1.2)
        assert_array_almost_equal(grid.gamma_fcx, np.ones(
            (8, 8)) * np.sqrt(0.4**2 + 1), decimal=12)
        assert_array_almost_equal(grid.gamma_fcy, np.ones(
            (8, 8)) * np.sqrt(0.5**2 + 1), decimal=12)

        Rx = np.array([[-0.4 / np.sqrt(0.4**2 + 1), 1 / np.sqrt(0.4**2 + 1)],
                       [-1 / np.sqrt(0.4**2 + 1), -0.4 / np.sqrt(0.4**2 + 1)]])

        Ry = np.array([[0.5 / np.sqrt(0.5**2 + 1), 1 / np.sqrt(0.5**2 + 1)],
                       [-1 / np.sqrt(0.5**2 + 1), 0.5 / np.sqrt(0.5**2 + 1)]])

        assert_array_almost_equal(grid.R_fcx(
            2, 0, 1)[0, 0, :, :], Rx, decimal=12)
        assert_array_almost_equal(grid.R_fcy(
            2, 0, 1)[2, 4, :, :], Ry, decimal=12)

    def test_polar(self):
        """
        Test mapped grid class for polar coordinates.
        """

        def map(myg):
            return sympy.Matrix([x * sympy.cos(y), x * sympy.sin(y)])

        grid = mapped.MappedGrid2d(map, 4, 4, ng=2, xmin=0.1)

        xs = grid.x2d * np.cos(grid.y2d)
        ys = grid.x2d * np.sin(grid.y2d)

        x_map = sympy.lambdify((x, y), map(grid)[0])(grid.x2d, grid.y2d)
        y_map = sympy.lambdify((x, y), map(grid)[1])(grid.x2d, grid.y2d)

        assert_array_almost_equal(xs, x_map, decimal=12)
        assert_array_almost_equal(ys, y_map, decimal=12)

        kappa = np.array([[0.23503376, 0.23503376, 0.23503376, 0.23503376, 0.23503376, 0.23503376, 0.23503376, 0.23503376],
                          [0.0123702,  0.0123702,  0.0123702,  0.0123702,  0.0123702,  0.0123702,
                           0.0123702,  0.0123702],
                          [0.21029337, 0.21029337, 0.21029337, 0.21029337, 0.21029337, 0.21029337,
                           0.21029337, 0.21029337],
                          [0.43295693, 0.43295693, 0.43295693, 0.43295693, 0.43295693, 0.43295693,
                           0.43295693, 0.43295693],
                          [0.65562049, 0.65562049, 0.65562049, 0.65562049, 0.65562049, 0.65562049,
                           0.65562049, 0.65562049],
                          [0.87828406, 0.87828406, 0.87828406, 0.87828406, 0.87828406, 0.87828406,
                           0.87828406, 0.87828406],
                          [1.10094762, 1.10094762, 1.10094762, 1.10094762, 1.10094762, 1.10094762,
                           1.10094762, 1.10094762],
                          [1.32361118, 1.32361118, 1.32361118, 1.32361118, 1.32361118, 1.32361118,
                           1.32361118, 1.32361118]])
        assert_array_almost_equal(grid.kappa, kappa)

        gamma_fcx = np.array([[0.34908925, 0.34908925, 0.34908925, 0.34908925, 0.34908925, 0.34908925,  0.34908925, 0.34908925],
                              [0.12467473, 0.12467473, 0.12467473, 0.12467473,
                                  0.12467473, 0.12467473, 0.12467473, 0.12467473],
                              [0.09973979, 0.09973979, 0.09973979, 0.09973979, 0.09973979, 0.09973979,
                               0.09973979, 0.09973979],
                              [0.32415431, 0.32415431, 0.32415431, 0.32415431, 0.32415431, 0.32415431,
                               0.32415431, 0.32415431],
                              [0.54856883, 0.54856883, 0.54856883, 0.54856883, 0.54856883, 0.54856883,
                               0.54856883, 0.54856883],
                              [0.77298335, 0.77298335, 0.77298335, 0.77298335, 0.77298335, 0.77298335,
                               0.77298335, 0.77298335],
                              [0.99739787, 0.99739787, 0.99739787, 0.99739787, 0.99739787, 0.99739787,
                               0.99739787, 0.99739787],
                              [1.22181239, 1.22181239, 1.22181239, 1.22181239, 1.22181239, 1.22181239,
                               1.22181239, 1.22181239]]
                             )
        assert_array_almost_equal(grid.gamma_fcx, gamma_fcx)

        gamma_fcy = np.ones_like(xs)
        assert_array_almost_equal(grid.gamma_fcy, gamma_fcy, decimal=12)

        Rx = lambda xx, yy: np.array([[np.cos(yy), -np.sin(yy)],
                                      [np.sin(yy), np.cos(yy)]])

        Ry = lambda xx, yy: np.array([[np.cos(yy), np.sin(yy)],
                                      [-np.sin(yy), np.cos(yy)]])

        R_fcx = Rx(grid.x2d[4, 5] - 0.5 * grid.dx, grid.y2d[4, 5])
        R_fcy = Ry(grid.x2d[3, 2], grid.y2d[3, 2] - 0.5 * grid.dy)

        assert_array_almost_equal(grid.R_fcx(
            2, 0, 1)[4, 5, :, :], R_fcx, decimal=12)
        assert_array_almost_equal(grid.R_fcy(
            2, 0, 1)[3, 2, :, :], R_fcy, decimal=12)


class TestMappedCellCenterData2d(object):

    @classmethod
    def setup_class(cls):
        """ this is run once for each class before any tests """
        pass

    @classmethod
    def teardown_class(cls):
        """ this is run once for each class after all tests """
        pass

    def setup_method(self):
        """ this is run before each test """
        self.g = mapped.MappedGrid2d(4, 6, ng=2, ymax=1.5)

    def teardown_method(self):
        """ this is run after each test """
        self.g = None
