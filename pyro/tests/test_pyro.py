import numpy as np
from numpy.testing import assert_array_equal

from pyro.pyro_sim import Pyro


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
        pass

    def teardown_method(self):
        """ this is run after each test """
        pass

    def test_pyro_class(self):
        """
        Check that the class sets everything up ok.
        """

        pyro_sim = Pyro("advection")
        pyro_sim.initialize_problem("test")

        assert pyro_sim.solver_name == "advection"

        assert pyro_sim.sim.problem_name == "test"

        dens = pyro_sim.sim.cc_data.get_var('density')

        assert_array_equal(dens, np.ones_like(dens))

    def test_run_sim(self):
        """
        Check that the class can run a simple simulation.
        """

        pyro_sim = Pyro("advection")

        # 5 steps
        inputs_dict = {"driver.max_steps": 5,
                       "vis.dovis": 0,
                       "mesh.nx": 8,
                       "mesh.ny": 8,
                       "driver.fix_dt": 0.2}

        pyro_sim.initialize_problem("test", inputs_dict=inputs_dict)

        pyro_sim.run_sim()

        dens = pyro_sim.sim.cc_data.get_var('density')

        assert_array_equal(dens, np.ones_like(dens))

        assert pyro_sim.sim.cc_data.t == 1
