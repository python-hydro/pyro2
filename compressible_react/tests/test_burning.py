import compressible_react.burning as burning
from util import runparams
from numpy.testing import assert_array_equal, assert_array_almost_equal
import numpy as np
import compressible_react.simulation as sn


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
        self.rp.params["eos.cv"] = 0.5
        self.rp.params["eos.cp"] = 1
        self.rp.params["eos.gamma"] = 1.4
        self.rp.params["compressible.grav"] = 1.0
        self.rp.params["mesh.nx"] = 8
        self.rp.params["mesh.ny"] = 8
        self.rp.params["particles.do_particles"] = 0

        self.sim = sn.Simulation("compressible_react", "test", self.rp)

        self.rp = self.sim.rp

    def teardown_method(self):
        """ this is run after each test """
        self.rp = None
        self.sim = None

    def test_null_network(self):
        """
        Test the null network
        """

        self.rp.params["network.network_type"] = "null"
        self.sim.initialize()

        ntk = self.sim.network

        assert ntk.nspec == 0
        assert ntk.spec_names is None

        ntk.initialize(self.rp)

        assert ntk.Cv == self.rp.params["eos.cv"]
        assert ntk.Cp == self.rp.params["eos.cp"]

        H, omega_dot = ntk.energy_and_species_creation(self.sim.cc_data)

        assert_array_equal(H, np.zeros_like(H))
        assert_array_equal(omega_dot, np.zeros_like(omega_dot))

    def test_powerlaw_network(self):
        """
        Test the powerlaw network
        """

        self.rp.params["network.network_type"] = "powerlaw"
        self.rp.params["network.specific_q_burn"] = 10
        self.rp.params["network.f_act"] = 1
        self.rp.params["network.t_burn_ref"] = 2
        self.rp.params["network.rho_burn_ref"] = 3
        self.rp.params["network.rtilde"] = 4
        self.rp.params["network.nu"] = 5

        self.sim.initialize()

        ntk = self.sim.network

        assert ntk.nspec == 3
        assert ntk.nspec_evolve == 2
        assert_array_equal(ntk.A_ion, [2, 4, 8])

        rate = 125 / 24

        myg = self.sim.cc_data.grid
        omega_dot_correct = myg.scratch_array(nvar=2)
        omega_dot_correct[:, :, 0] = -rate / 2
        omega_dot_correct[:, :, 1] = rate / 4

        E_nuc_correct = myg.scratch_array()
        E_nuc_correct[:, :] = rate * self.rp.params["network.specific_q_burn"]

        E_nuc, omega_dot = ntk.energy_and_species_creation(self.sim.cc_data)

        assert_array_almost_equal(omega_dot, omega_dot_correct)
        assert_array_almost_equal(E_nuc, E_nuc_correct)

    def test_k_th(self):
        """
        Test the conductivity
        """
        self.rp.params["network.network_type"] = "null"
        self.sim.initialize()

        myd = self.sim.cc_data
        myg = myd.grid

        k = burning.k_th(myd, 1, 30, 1)

        assert_array_equal(myg.scratch_array() + 30, k)

        temp = myg.scratch_array() + 1
        k_correct = myg.scratch_array() + 1.512106667e-4

        assert_array_almost_equal(k_correct, burning.k_th(myd, temp, 2, 0))
