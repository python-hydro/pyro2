# unit tests for the patch
import numpy as np
from numpy.testing import assert_array_equal

import pyro.mesh.boundary as bnd
import pyro.mesh.patch as patch


# Grid2d tests
class TestGrid2d(object):
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
        self.g = patch.Grid2d(4, 6, ng=2, ymax=1.5)

    def teardown_method(self):
        """ this is run after each test """
        self.g = None

    def test_dx_dy(self):
        assert self.g.dx == 0.25
        assert self.g.dy == 0.25

    def test_grid_coords(self):
        assert_array_equal(self.g.x[self.g.ilo:self.g.ihi+1],
                           np.array([0.125, 0.375, 0.625, 0.875]))
        assert_array_equal(self.g.y[self.g.jlo:self.g.jhi+1],
                           np.array([0.125, 0.375, 0.625, 0.875, 1.125, 1.375]))

    def test_grid_2d_coords(self):
        assert_array_equal(self.g.x, self.g.x2d[:, self.g.jc])
        assert_array_equal(self.g.y, self.g.y2d[self.g.ic, :])

    def test_scratch_array(self):
        q = self.g.scratch_array()
        assert q.shape == (self.g.qx, self.g.qy)

    def test_coarse_like(self):
        q = self.g.coarse_like(2)
        assert q.qx == 2*self.g.ng + self.g.nx//2
        assert q.qy == 2*self.g.ng + self.g.ny//2

    def test_fine_like(self):
        q = self.g.fine_like(2)
        assert q.qx == 2*self.g.ng + 2*self.g.nx
        assert q.qy == 2*self.g.ng + 2*self.g.ny

    def test_norm(self):
        q = self.g.scratch_array()
        # there are 24 elements, the norm L2 norm is
        # sqrt(dx*dy*24)
        q.v()[:, :] = np.array([[1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1]])

        assert q.norm() == np.sqrt(24*self.g.dx*self.g.dy)

    def test_equality(self):
        g2 = patch.Grid2d(2, 5, ng=1)
        assert g2 != self.g


# CellCenterData2d tests
class TestCellCenterData2d(object):
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
        nx = 8
        ny = 8
        self.g = patch.Grid2d(nx, ny, ng=2, xmax=1.0, ymax=1.0)
        self.d = patch.CellCenterData2d(self.g, dtype=np.int)

        bco = bnd.BC(xlb="outflow", xrb="outflow",
                     ylb="outflow", yrb="outflow")
        self.d.register_var("a", bco)
        self.d.register_var("b", bco)
        self.d.create()

    def teardown_method(self):
        """ this is run after each test """
        self.g = None
        self.d = None

    def test_zeros(self):

        a = self.d.get_var("a")
        a[:, :] = 1.0

        self.d.zero("a")
        assert np.all(a.v() == 0.0)

    def test_aux(self):
        self.d.set_aux("ftest", 1.0)
        self.d.set_aux("stest", "this was a test")

        assert self.d.get_aux("ftest") == 1.0
        assert self.d.get_aux("stest") == "this was a test"

    def test_gets(self):
        aname = self.d.get_var("a")
        aname[:, :] = np.random.rand(aname.shape[0], aname.shape[1])

        aindex = self.d.get_var_by_index(0)

        assert_array_equal(aname, aindex)

    def test_min_and_max(self):
        a = self.d.get_var("a")
        a.v()[:, :] = np.arange(self.g.nx*self.g.ny).reshape(self.g.nx, self.g.ny) + 1
        assert self.d.min("a") == 1.0
        assert self.d.max("a") == 64.0

    def test_restrict(self):
        a = self.d.get_var("a")
        a.v()[:, :] = np.arange(self.g.nx*self.g.ny).reshape(self.g.nx, self.g.ny) + 1

        c = self.d.restrict("a")

        # restriction should be conservative, so compare the volume-weighted sums
        assert np.sum(a.v()) == 4.0*np.sum(c.v())

    def test_prolong(self):
        a = self.d.get_var("a")
        a.v()[:, :] = np.arange(self.g.nx*self.g.ny).reshape(self.g.nx, self.g.ny) + 1

        f = self.d.prolong("a")

        # prologation should be conservative, so compare the volume-weighted sums
        assert 4.0*np.sum(a.v()) == np.sum(f.v())

    def test_zero(self):
        a = self.d.get_var("a")
        a.v()[:, :] = np.arange(self.g.nx*self.g.ny).reshape(self.g.nx, self.g.ny) + 1

        self.d.zero("a")
        assert self.d.min("a") == 0.0 and self.d.max("a") == 0.0


def test_bcs():

    myg = patch.Grid2d(4, 4, ng=2, xmax=1.0, ymax=1.0)
    myd = patch.CellCenterData2d(myg, dtype=np.int)

    bco = bnd.BC(xlb="outflow", xrb="outflow",
                 ylb="outflow", yrb="outflow")
    myd.register_var("outflow", bco)

    bcp = bnd.BC(xlb="periodic", xrb="periodic",
                 ylb="periodic", yrb="periodic")
    myd.register_var("periodic", bcp)

    bcre = bnd.BC(xlb="reflect-even", xrb="reflect-even",
                  ylb="reflect-even", yrb="reflect-even")
    myd.register_var("reflect-even", bcre)

    bcro = bnd.BC(xlb="reflect-odd", xrb="reflect-odd",
                  ylb="reflect-odd", yrb="reflect-odd")
    myd.register_var("reflect-odd", bcro)

    myd.create()

    a = myd.get_var("outflow")
    a.v()[:, :] = np.fromfunction(lambda i, j: i+10*j+1, (4, 4), dtype=int)

    b = myd.get_var("periodic")
    c = myd.get_var("reflect-even")
    d = myd.get_var("reflect-odd")

    b[:, :] = a[:, :]
    c[:, :] = a[:, :]
    d[:, :] = a[:, :]

    myd.fill_BC("outflow")
    # left ghost
    assert_array_equal(a[myg.ilo-1, myg.jlo:myg.jhi+1], np.array([1, 11, 21, 31]))
    # right ghost
    assert_array_equal(a[myg.ihi+1, myg.jlo:myg.jhi+1], np.array([4, 14, 24, 34]))
    # bottom ghost
    assert_array_equal(a[myg.ilo:myg.ihi+1, myg.jlo-1], np.array([1, 2, 3, 4]))
    # top ghost
    assert_array_equal(a[myg.ilo:myg.ihi+1, myg.jhi+1], np.array([31, 32, 33, 34]))

    myd.fill_BC("periodic")
    # x-boundaries
    assert_array_equal(b[myg.ilo-1, myg.jlo:myg.jhi+1],
                       b[myg.ihi, myg.jlo:myg.jhi+1])
    assert_array_equal(b[myg.ilo, myg.jlo:myg.jhi+1],
                       b[myg.ihi+1, myg.jlo:myg.jhi+1])
    # y-boundaries
    assert_array_equal(b[myg.ilo:myg.ihi+1, myg.jlo-1],
                       b[myg.ilo:myg.ihi+1, myg.jhi])
    assert_array_equal(b[myg.ilo:myg.ihi+1, myg.jlo],
                       b[myg.ilo:myg.ihi+1, myg.jhi+1])

    myd.fill_BC("reflect-even")
    # left -- we'll check 2 ghost cells here -- now we use flipud here
    # because our 'x' is the row index
    # left
    assert_array_equal(c[myg.ilo:myg.ilo+2, myg.jlo:myg.ihi+1],
                       np.flipud(c[myg.ilo-2:myg.ilo, myg.jlo:myg.jhi+1]))
    # right
    assert_array_equal(c[myg.ihi-1:myg.ihi+1, myg.jlo:myg.jhi+1],
                       np.flipud(c[myg.ihi+1:myg.ihi+3, myg.jlo:myg.jhi+1]))

    # bottom
    assert_array_equal(c[myg.ilo:myg.ihi+1, myg.jlo:myg.jlo+2],
                       np.fliplr(c[myg.ilo:myg.ihi+1, myg.jlo-2:myg.jlo]))
    # top
    assert_array_equal(c[myg.ilo:myg.ihi+1, myg.jhi-1:myg.jhi+1],
                       np.fliplr(c[myg.ilo:myg.ihi+1, myg.jhi+1:myg.jhi+3]))

    myd.fill_BC("reflect-odd")
    # left -- we'll check 2 ghost cells here -- now we use flipud here
    # because our 'x' is the row index
    # left
    assert_array_equal(d[myg.ilo:myg.ilo+2, myg.jlo:myg.ihi+1],
                       -np.flipud(d[myg.ilo-2:myg.ilo, myg.jlo:myg.jhi+1]))
    # right
    assert_array_equal(d[myg.ihi-1:myg.ihi+1, myg.jlo:myg.jhi+1],
                       -np.flipud(d[myg.ihi+1:myg.ihi+3, myg.jlo:myg.jhi+1]))

    # bottom
    assert_array_equal(d[myg.ilo:myg.ihi+1, myg.jlo:myg.jlo+2],
                       -np.fliplr(d[myg.ilo:myg.ihi+1, myg.jlo-2:myg.jlo]))
    # top
    assert_array_equal(d[myg.ilo:myg.ihi+1, myg.jhi-1:myg.jhi+1],
                       -np.fliplr(d[myg.ilo:myg.ihi+1, myg.jhi+1:myg.jhi+3]))
