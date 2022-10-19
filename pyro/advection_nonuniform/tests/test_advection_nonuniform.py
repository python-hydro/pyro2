import pyro.advection_nonuniform.simulation as sn
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

        self.rp.params["mesh.nx"] = 8
        self.rp.params["mesh.ny"] = 8
        self.rp.params["particles.do_particles"] = 0

        self.sim = sn.Simulation("advection_nonuniform", "test", self.rp)
        self.sim.initialize()

    def teardown_method(self):
        """ this is run after each test """
        self.rp = None
        self.sim = None

    def test_initializationst(self):
        dens = self.sim.cc_data.get_var("density")
        u = self.sim.cc_data.get_var("x-velocity")
        v = self.sim.cc_data.get_var("y-velocity")

        assert dens.min() == 1.0 and dens.max() == 1.0
        assert u.min() == 1.0 and u.max() == 1.0
        assert v.min() == 1.0 and v.max() == 1.0
