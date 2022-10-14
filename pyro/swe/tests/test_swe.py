import numpy as np
from numpy.testing import assert_array_equal

import pyro.swe.simulation as sn
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

        self.rp.params["swe.grav"] = 1.0

        self.sim = sn.Simulation("swe", "test", self.rp)
        self.sim.initialize()

    def teardown_method(self):
        """ this is run after each test """
        self.rp = None
        self.sim = None

    def test_initializationst(self):
        h = self.sim.cc_data.get_var("height")
        assert h.min() == 1.0 and h.max() == 1.0

    def test_prim(self):

        # U -> q
        g = self.sim.cc_data.get_aux("g")
        q = sn.cons_to_prim(self.sim.cc_data.data, g, self.sim.ivars, self.sim.cc_data.grid)

        # q -> U
        U = sn.prim_to_cons(q, g, self.sim.ivars, self.sim.cc_data.grid)
        assert_array_equal(U, self.sim.cc_data.data)

    def test_derives(self):

        g = self.sim.cc_data.get_aux("g")
        cs = self.sim.cc_data.get_var("soundspeed")
        assert np.all(cs == np.sqrt(g))
