# unit tests for the patch
import mesh.boundary as bnd
import mesh.patch as patch
import mesh.array_indexer as ai
import numpy as np

from numpy.testing import assert_array_equal
from nose.tools import assert_equal, with_setup

# utilities
def test_buf_split():
    assert_array_equal(ai._buf_split(2), [2, 2, 2, 2])
    assert_array_equal(ai._buf_split((2, 3)), [2, 3, 2, 3])


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

    def setup(self):
        """ this is run before each test """
        self.g = patch.Grid2d(4, 6, ng=2, ymax=1.5)

    def teardown(self):
        """ this is run after each test """
        self.g = None

    def test_dx_dy(self):
        assert_equal(self.g.dx, 0.25)
        assert_equal(self.g.dy, 0.25)

    def test_grid_coords(self):
        assert_array_equal(self.g.x[self.g.ilo:self.g.ihi+1], 
                           np.array([0.125, 0.375, 0.625, 0.875]))
        assert_array_equal(self.g.y[self.g.jlo:self.g.jhi+1], 
                           np.array([0.125, 0.375, 0.625, 0.875, 1.125, 1.375]))

    def test_grid_2d_coords(self):
        assert_array_equal(self.g.x, self.g.x2d[:,self.g.jc])
        assert_array_equal(self.g.y, self.g.y2d[self.g.ic,:])

    def test_course_like(self):
        c = self.g.coarse_like(2)

        assert_equal(c.nx, 2)
        assert_equal(c.ny, 3)


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

    def setup(self):
        """ this is run before each test """
        nx = 8
        ny = 8
        self.g = patch.Grid2d(nx, ny, ng = 2, xmax=1.0, ymax=1.0)
        self.d = patch.CellCenterData2d(self.g, dtype=np.int)

        bco = bnd.BC(xlb="outflow", xrb="outflow",
                     ylb="outflow", yrb="outflow")
        self.d.register_var("a", bco)
        self.d.register_var("b", bco)
        self.d.create()

    def teardown(self):
        """ this is run after each test """
        self.g = None
        self.d = None

    def test_zeros(self):
    
        a = self.d.get_var("a")
        a[:,:] = 1.0

        self.d.zero("a")
        assert_equal(np.all(a.v() == 0.0), True)

    def test_aux(self):
        self.d.set_aux("ftest", 1.0)
        self.d.set_aux("stest", "this was a test")

        assert_equal(self.d.get_aux("ftest"), 1.0)
        assert_equal(self.d.get_aux("stest"), "this was a test")
    

def test_bcs():

    myg = patch.Grid2d(4,4, ng = 2, xmax=1.0, ymax=1.0)
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
    a.v()[:,:] = np.fromfunction(lambda i, j: i+10*j+1, (4,4), dtype=int)

    b = myd.get_var("periodic")
    c = myd.get_var("reflect-even")
    d = myd.get_var("reflect-odd")

    b[:,:] = a[:,:]
    c[:,:] = a[:,:]
    d[:,:] = a[:,:]

    myd.fill_BC("outflow")
    # left ghost
    assert_array_equal(a[myg.ilo-1,myg.jlo:myg.jhi+1], np.array([ 1, 11, 21, 31]))
    # right ghost
    assert_array_equal(a[myg.ihi+1,myg.jlo:myg.jhi+1], np.array([ 4, 14, 24, 34]))
    # bottom ghost
    assert_array_equal(a[myg.ilo:myg.ihi+1,myg.jlo-1], np.array([ 1, 2, 3, 4]))
    # top ghost
    assert_array_equal(a[myg.ilo:myg.ihi+1,myg.jhi+1], np.array([31, 32, 33, 34]))

    myd.fill_BC("periodic")
    # x-boundaries
    assert_array_equal(b[myg.ilo-1,myg.jlo:myg.jhi+1], 
                       b[myg.ihi,myg.jlo:myg.jhi+1])
    assert_array_equal(b[myg.ilo,myg.jlo:myg.jhi+1], 
                       b[myg.ihi+1,myg.jlo:myg.jhi+1])
    # y-boundaries
    assert_array_equal(b[myg.ilo:myg.ihi+1,myg.jlo-1],
                       b[myg.ilo:myg.ihi+1,myg.jhi])
    assert_array_equal(b[myg.ilo:myg.ihi+1,myg.jlo],
                       b[myg.ilo:myg.ihi+1,myg.jhi+1])

    myd.fill_BC("reflect-even")
    # left -- we'll check 2 ghost cells here -- now we use flipud here
    # because our 'x' is the row index
    # left
    assert_array_equal(c[myg.ilo:myg.ilo+2,myg.jlo:myg.ihi+1],
                       np.flipud(c[myg.ilo-2:myg.ilo,myg.jlo:myg.jhi+1]))
    # right
    assert_array_equal(c[myg.ihi-1:myg.ihi+1,myg.jlo:myg.jhi+1],
                       np.flipud(c[myg.ihi+1:myg.ihi+3,myg.jlo:myg.jhi+1]))
    
    # bottom
    assert_array_equal(c[myg.ilo:myg.ihi+1,myg.jlo:myg.jlo+2],
                       np.fliplr(c[myg.ilo:myg.ihi+1,myg.jlo-2:myg.jlo]))
    # top
    assert_array_equal(c[myg.ilo:myg.ihi+1,myg.jhi-1:myg.jhi+1],
                       np.fliplr(c[myg.ilo:myg.ihi+1,myg.jhi+1:myg.jhi+3]))
    
    myd.fill_BC("reflect-odd")
    # left -- we'll check 2 ghost cells here -- now we use flipud here
    # because our 'x' is the row index
    # left
    assert_array_equal(d[myg.ilo:myg.ilo+2,myg.jlo:myg.ihi+1],
                       -np.flipud(d[myg.ilo-2:myg.ilo,myg.jlo:myg.jhi+1]))
    # right
    assert_array_equal(d[myg.ihi-1:myg.ihi+1,myg.jlo:myg.jhi+1],
                       -np.flipud(d[myg.ihi+1:myg.ihi+3,myg.jlo:myg.jhi+1]))
    
    # bottom
    assert_array_equal(d[myg.ilo:myg.ihi+1,myg.jlo:myg.jlo+2],
                       -np.fliplr(d[myg.ilo:myg.ihi+1,myg.jlo-2:myg.jlo]))
    # top
    assert_array_equal(d[myg.ilo:myg.ihi+1,myg.jhi-1:myg.jhi+1],
                       -np.fliplr(d[myg.ilo:myg.ihi+1,myg.jhi+1:myg.jhi+3]))



# ArrayIndexer tests
def test_indexer():
    g = patch.Grid2d(2,3, ng=2)
    a = g.scratch_array()

    a[:,:] = np.arange(g.qx*g.qy).reshape(g.qx, g.qy)
    
    assert_array_equal(a.v(), np.array([[16., 17., 18.], [23., 24., 25.]]))

    assert_array_equal(a.ip(1), np.array([[23., 24., 25.], [30., 31., 32.]]))
    assert_array_equal(a.ip(-1), np.array([[9., 10., 11.], [16., 17., 18.]]))

    assert_array_equal(a.jp(1), np.array([[17., 18., 19.], [24., 25., 26.]]))
    assert_array_equal(a.jp(-1), np.array([[15., 16., 17.], [ 22., 23., 24.]]))

    assert_array_equal(a.ip_jp(1, 1), np.array([[24., 25., 26.], [ 31., 32., 33.]]))

def test_is_symmetric():
    g = patch.Grid2d(4, 3, ng=0)
    a = g.scratch_array()

    a[:,0] = [1, 2, 2, 1]
    a[:,1] = [2, 4, 4, 2]
    a[:,2] = [1, 2, 2, 1]

    assert_equal(a.is_symmetric(), True)


def test_is_asymmetric():
    g = patch.Grid2d(4, 3, ng=0)
    a = g.scratch_array()

    a[:,0] = [-1, -2, 2, 1]
    a[:,1] = [-2, -4, 4, 2]
    a[:,2] = [-1, -2, 2, 1]

    assert_equal(a.is_asymmetric(), True)



