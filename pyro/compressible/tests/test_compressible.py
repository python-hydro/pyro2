import numpy as np
import pytest
from numpy.testing import assert_array_equal

import pyro.compressible.simulation as sn
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

        self.rp.params["eos.gamma"] = 1.4
        self.rp.params["compressible.grav"] = 1.0

        self.sim = sn.Simulation("compressible", "test", self.rp)
        self.sim.initialize()

    def teardown_method(self):
        """ this is run after each test """
        self.rp = None
        self.sim = None

    def test_initializationst(self):
        dens = self.sim.cc_data.get_var("density")
        assert dens.min() == 1.0 and dens.max() == 1.0

        ener = self.sim.cc_data.get_var("energy")
        assert ener.min() == 2.5 and ener.max() == 2.5

    def test_prim(self):

        # U -> q
        gamma = self.sim.cc_data.get_aux("gamma")
        q = sn.cons_to_prim(self.sim.cc_data.data, gamma, self.sim.ivars, self.sim.cc_data.grid)

        assert q[:, :, self.sim.ivars.ip].min() == pytest.approx(1.0) and \
               q[:, :, self.sim.ivars.ip].max() == pytest.approx(1.0)

        # q -> U
        U = sn.prim_to_cons(q, gamma, self.sim.ivars, self.sim.cc_data.grid)
        assert_array_equal(U, self.sim.cc_data.data)

    def test_derives(self):

        gamma = self.sim.cc_data.get_aux("gamma")
        cs = self.sim.cc_data.get_var("soundspeed")
        assert np.all(cs == np.sqrt(gamma))
