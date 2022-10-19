import pyro.mesh.boundary as bnd
import pyro.mesh.patch as patch
import pyro.simulation_null as sn
from pyro.util import runparams


class TestSimulation(object):

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
        self.rp = runparams.RuntimeParameters()
        self.rp.params["driver.tmax"] = 1.0
        self.rp.params["driver.max_steps"] = 100
        self.rp.params["driver.init_tstep_factor"] = 0.5
        self.rp.params["driver.max_dt_change"] = 1.2
        self.rp.params["driver.fix_dt"] = -1.0

        self.sim = sn.NullSimulation("test", "test", self.rp)

        myg = patch.Grid2d(8, 16)
        myd = patch.CellCenterData2d(myg)
        bc = bnd.BC()
        myd.register_var("a", bc)
        myd.create()

        self.sim.cc_data = myd

    def teardown_method(self):
        """ this is run after each test """
        self.rp = None
        self.sim = None

    def test_finished_n(self):
        self.sim.n = 1000
        assert self.sim.finished()

    def test_finished_t(self):
        self.sim.cc_data.t = 2.0
        assert self.sim.finished()

    def test_compute_timestep(self):

        # set a dt and n = 0, then init_tstep_factor should kick in
        self.sim.dt = 2.0
        self.sim.n = 0
        self.sim.compute_timestep()
        assert self.sim.dt == 1.0

        # now set dt_old and a new dt and see if the max_dt_change kicks in
        self.sim.n = 1.0
        self.sim.dt_old = 1.0
        self.sim.dt = 2.0
        self.sim.compute_timestep()
        assert self.sim.dt == 1.2

        # now test what happens if we go over tmax
        self.sim.cc_data.t = 0.75
        self.dt = 0.5
        self.sim.compute_timestep()
        assert self.sim.dt == 0.25


def test_grid_setup():

    rp = runparams.RuntimeParameters()
    rp.params["mesh.nx"] = 8
    rp.params["mesh.ny"] = 16
    rp.params["mesh.xmin"] = 0.0
    rp.params["mesh.xmax"] = 1.0
    rp.params["mesh.ymin"] = 0.0
    rp.params["mesh.ymax"] = 2.0

    g = sn.grid_setup(rp)

    assert g.nx == 8
    assert g.ny == 16
    assert g.dx == 1.0/8
    assert g.dy == 1.0/8
