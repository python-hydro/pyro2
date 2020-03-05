from util import runparams
import advection.simulation as sn


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

        self.sim = sn.Simulation("advection", "test", self.rp)
        self.sim.initialize()

    def teardown_method(self):
        """ this is run after each test """
        self.rp = None
        self.sim = None

    def test_initializationst(self):
        dens = self.sim.cc_data.get_var("density")
        assert dens.min() == 1.0 and dens.max() == 1.0
