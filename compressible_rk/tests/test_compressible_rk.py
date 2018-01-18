import numpy as np
from numpy.testing import assert_array_equal

from util import runparams
import compressible_rk.simulation as sn
import mesh.patch as patch
import mesh.boundary as bnd
import pytest

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
