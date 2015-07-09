# unit tests for the patch
import mesh.patch as patch
import numpy as np

from numpy.testing import assert_array_equal
from nose.tools import assert_equal

import util.testing_help as th

# utilities
@th.with_named_setup(th.setup_func, th.teardown_func)
def test_buf_split():
    assert_array_equal(patch._buf_split(2), [2, 2, 2, 2])
    assert_array_equal(patch._buf_split((2, 3)), [2, 3, 2, 3])


# Grid2d tests

@th.with_named_setup(th.setup_func, th.teardown_func)
def test_dx_dy():
    g = patch.Grid2d(4, 6, ng=2, ymax=1.5)

    assert_equal(g.dx, 0.25)
    assert_equal(g.dy, 0.25)

@th.with_named_setup(th.setup_func, th.teardown_func)
def test_grid_coords():
    g = patch.Grid2d(4, 6, ng=2, ymax=1.5)

    assert_array_equal(g.x[g.ilo:g.ihi+1], np.array([0.125, 0.375, 0.625, 0.875]))
    assert_array_equal(g.y[g.jlo:g.jhi+1], np.array([0.125, 0.375, 0.625, 0.875, 1.125, 1.375]))

@th.with_named_setup(th.setup_func, th.teardown_func)
def test_grid_2d_coords():
    g = patch.Grid2d(4, 6, ng=2, ymax=1.5)

    assert_array_equal(g.x, g.x2d[:,g.jc])
    assert_array_equal(g.y, g.y2d[g.ic,:])

@th.with_named_setup(th.setup_func, th.teardown_func)
def test_course_like():
    g = patch.Grid2d(4, 6, ng=2, ymax=1.5)
    c = g.coarse_like(2)

    assert_equal(c.nx, 2)
    assert_equal(c.ny, 3)


# CellCenterData2d tests
@th.with_named_setup(th.setup_func, th.teardown_func)
def test_bcs():

    myg = patch.Grid2d(4,4, ng = 2, xmax=1.0, ymax=1.0)
    myd = patch.CellCenterData2d(myg, dtype=np.int)

    bco = patch.BCObject(xlb="outflow", xrb="outflow",
                         ylb="outflow", yrb="outflow")
    myd.register_var("outflow", bco)

    bcp = patch.BCObject(xlb="periodic", xrb="periodic",
                         ylb="periodic", yrb="periodic")
    myd.register_var("periodic", bcp)

    bcre = patch.BCObject(xlb="reflect-even", xrb="reflect-even",
                          ylb="reflect-even", yrb="reflect-even")
    myd.register_var("reflect-even", bcre)

    bcro = patch.BCObject(xlb="reflect-odd", xrb="reflect-odd",
                          ylb="reflect-odd", yrb="reflect-odd")
    myd.register_var("reflect-odd", bcro)

    myd.create()


    a = myd.get_var("outflow")
    a.v()[:,:] = np.fromfunction(lambda i, j: i+10*j+1, (4,4), dtype=int)

    b = myd.get_var("periodic")
    c = myd.get_var("reflect-even")
    d = myd.get_var("reflect-odd")

    b.d[:,:] = a.d[:,:]
    c.d[:,:] = a.d[:,:]
    d.d[:,:] = a.d[:,:]

    myd.fill_BC("outflow")
    # left ghost
    assert_array_equal(a.d[myg.ilo-1,myg.jlo:myg.jhi+1], np.array([ 1, 11, 21, 31]))
    # right ghost
    assert_array_equal(a.d[myg.ihi+1,myg.jlo:myg.jhi+1], np.array([ 4, 14, 24, 34]))
    # bottom ghost
    assert_array_equal(a.d[myg.ilo:myg.ihi+1,myg.jlo-1], np.array([ 1, 2, 3, 4]))
    # top ghost
    assert_array_equal(a.d[myg.ilo:myg.ihi+1,myg.jhi+1], np.array([31, 32, 33, 34]))

    myd.fill_BC("periodic")
    # x-boundaries
    assert_array_equal(b.d[myg.ilo-1,myg.jlo:myg.jhi+1], 
                       b.d[myg.ihi,myg.jlo:myg.jhi+1])
    assert_array_equal(b.d[myg.ilo,myg.jlo:myg.jhi+1], 
                       b.d[myg.ihi+1,myg.jlo:myg.jhi+1])
    # y-boundaries
    assert_array_equal(b.d[myg.ilo:myg.ihi+1,myg.jlo-1],
                       b.d[myg.ilo:myg.ihi+1,myg.jhi])
    assert_array_equal(b.d[myg.ilo:myg.ihi+1,myg.jlo],
                       b.d[myg.ilo:myg.ihi+1,myg.jhi+1])

    myd.fill_BC("reflect-even")
    # left -- we'll check 2 ghost cells here -- now we use flipud here
    # because our 'x' is the row index
    # left
    assert_array_equal(c.d[myg.ilo:myg.ilo+2,myg.jlo:myg.ihi+1],
                       np.flipud(c.d[myg.ilo-2:myg.ilo,myg.jlo:myg.jhi+1]))
    # right
    assert_array_equal(c.d[myg.ihi-1:myg.ihi+1,myg.jlo:myg.jhi+1],
                       np.flipud(c.d[myg.ihi+1:myg.ihi+3,myg.jlo:myg.jhi+1]))
    
    # bottom
    assert_array_equal(c.d[myg.ilo:myg.ihi+1,myg.jlo:myg.jlo+2],
                       np.fliplr(c.d[myg.ilo:myg.ihi+1,myg.jlo-2:myg.jlo]))
    # top
    assert_array_equal(c.d[myg.ilo:myg.ihi+1,myg.jhi-1:myg.jhi+1],
                       np.fliplr(c.d[myg.ilo:myg.ihi+1,myg.jhi+1:myg.jhi+3]))
    
    myd.fill_BC("reflect-odd")
    # left -- we'll check 2 ghost cells here -- now we use flipud here
    # because our 'x' is the row index
    # left
    assert_array_equal(d.d[myg.ilo:myg.ilo+2,myg.jlo:myg.ihi+1],
                       -np.flipud(d.d[myg.ilo-2:myg.ilo,myg.jlo:myg.jhi+1]))
    # right
    assert_array_equal(d.d[myg.ihi-1:myg.ihi+1,myg.jlo:myg.jhi+1],
                       -np.flipud(d.d[myg.ihi+1:myg.ihi+3,myg.jlo:myg.jhi+1]))
    
    # bottom
    assert_array_equal(d.d[myg.ilo:myg.ihi+1,myg.jlo:myg.jlo+2],
                       -np.fliplr(d.d[myg.ilo:myg.ihi+1,myg.jlo-2:myg.jlo]))
    # top
    assert_array_equal(d.d[myg.ilo:myg.ihi+1,myg.jhi-1:myg.jhi+1],
                       -np.fliplr(d.d[myg.ilo:myg.ihi+1,myg.jhi+1:myg.jhi+3]))





