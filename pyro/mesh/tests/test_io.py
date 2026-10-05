import numpy as np
from numpy.testing import assert_array_equal

import pyro.mesh.boundary as bnd
import pyro.util.io_pyro as io
from pyro import Pyro
from pyro.mesh import patch


def test_write_read():

    myg = patch.Grid2d(8, 6, ng=2, xmax=1.0, ymax=1.0)
    myd = patch.CellCenterData2d(myg)

    bco = bnd.BC(xlb="outflow", xrb="outflow",
                 ylb="outflow", yrb="outflow")
    myd.register_var("a", bco)

    myd.create()

    a = myd.get_var("a")
    a.v()[:, :] = np.arange(48).reshape(8, 6)

    myd.write("io_test")

    # now read it in
    nd = io.read("io_test")

    anew = nd.get_var("a")

    assert_array_equal(anew.v(), a.v())


def test_simulation_step_count_round_trip(tmp_path, monkeypatch):
    """Variable-name iteration must not overwrite the saved step counter."""
    monkeypatch.chdir(tmp_path)
    p = Pyro("advection")
    p.initialize_problem("test", inputs_dict={"mesh.nx": 8, "mesh.ny": 8,
        "driver.max_steps": 3, "io.force_final_output": 0})
    p.run_sim()
    p.sim.write("checkpoint")
    restored = io.read("checkpoint")
    assert restored.n == 3
    assert restored.cc_data.t == p.sim.cc_data.t
    assert_array_equal(restored.cc_data.get_var("density").v(), p.sim.cc_data.get_var("density").v())
