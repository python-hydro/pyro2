import pyro.diffusion.simulation as sim
from pyro.diffusion.problems import test
from pyro.util import runparams


class TestSimulation:

    @classmethod
    def setup_class(cls):
        """ this is run once for each class before any tests """

    @classmethod
    def teardown_class(cls):
        """ this is run once for each class after all tests """

    def setup_method(self):
        """ this is run before each test """
        self.rp = runparams.RuntimeParameters()

        self.rp.params["mesh.nx"] = 8
        self.rp.params["mesh.ny"] = 8

        self.sim = sim.Simulation("diffusion", "test", test.init_data, self.rp)
        self.sim.initialize()

    def teardown_method(self):
        """ this is run after each test """
        self.rp = None
        self.sim = None

    def test_initializationst(self):
        phi = self.sim.cc_data.get_var("phi")
        assert phi.min() == 1.0 and phi.max() == 1.0
