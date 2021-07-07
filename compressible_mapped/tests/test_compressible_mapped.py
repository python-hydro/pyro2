import numpy as np
from numpy.testing import assert_array_equal
import pytest
import sympy

from util import runparams
import compressible_mapped.simulation as sn


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

        self.sim = sn.Simulation("compressible_mapped", "test", self.rp)
        self.sim.initialize()

    def teardown_method(self):
        """ this is run after each test """
        self.rp = None
        self.sim = None

    def test_initialization_state(self):
        """
        Test state initialized properly
        """
        dens = self.sim.cc_data.get_var("density")
        assert dens.min() == 1.0 and dens.max() == 1.0

        ener = self.sim.cc_data.get_var("energy")
        assert ener.min() == 2.5 and ener.max() == 2.5

    def test_mapped_grid(self):

        myg = self.sim.cc_data.grid

        kappa = myg.scratch_array() + 2
        assert_array_equal(kappa, myg.kappa)

        gamma_fcx = np.ones_like(kappa) * 2
        assert_array_equal(gamma_fcx, myg.gamma_fcx)

        gamma_fcy = np.ones_like(kappa)
        assert_array_equal(gamma_fcy, myg.gamma_fcy)

        R_x = sympy.Matrix([[0, -1], [1, 0]])
        R_y = sympy.Matrix([[0, 1], [-1, 0]])

        assert_array_equal(myg.R_fcx(2, 0, 1)[3, 5], R_x)
        assert_array_equal(myg.R_fcy(2, 0, 1)[6, 0], R_y)
