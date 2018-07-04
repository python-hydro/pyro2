import particles.particles as particles
import mesh.patch as patch
from util import runparams
import numpy as np
from numpy.testing import assert_array_equal
from simulation_null import NullSimulation, grid_setup, bc_setup


def test_particle():
    n_particles = 5

    for n in range(n_particles):
        x, y = np.random.rand(2)
        p = particles.Particle(x, y)

        assert_array_equal(p.pos(), [x, y])
        assert_array_equal(p.velocity(), [0, 0])


def test_particles_random_gen():
    rp = runparams.RuntimeParameters()

    rp.params["mesh.nx"] = 8
    rp.params["mesh.ny"] = 8
    rp.params["particles.do_particles"] = 0
    rp.params["particles.n_particles"] = 50
    rp.params["particles.particle_generator"] = "random"

    # set up sim
    sim = NullSimulation("", "", rp)

    # set up grid
    my_grid = grid_setup(rp)
    my_data = patch.CellCenterData2d(my_grid)
    bc = bc_setup(rp)[0]
    my_data.create()
    sim.cc_data = my_data

    # seed random number generator
    np.random.seed(3287469)

    ps = particles.Particles(sim.cc_data, bc, rp)
    positions = set([(p.x, p.y) for p in ps.particles])

    assert len(ps.particles) == 50, "There should be 50 particles"

    # reseed random number generator
    np.random.seed(3287469)

    correct_positions = np.random.rand(50, 2)
    correct_positions = set([(x, y) for (x, y) in correct_positions])

    assert positions == correct_positions, "sets are not the same"


def test_particles_grid_gen():
    rp = runparams.RuntimeParameters()

    rp.params["mesh.nx"] = 8
    rp.params["mesh.ny"] = 8
    rp.params["particles.do_particles"] = 0
    rp.params["particles.n_particles"] = 50
    rp.params["particles.particle_generator"] = "grid"

    # set up sim
    sim = NullSimulation("", "", rp)

    # set up grid
    my_grid = grid_setup(rp)
    my_data = patch.CellCenterData2d(my_grid)
    bc = bc_setup(rp)[0]
    my_data.create()
    sim.cc_data = my_data

    ps = particles.Particles(sim.cc_data, bc, rp)

    assert len(ps.particles) == 49, "There should be 49 particles"

    positions = set([(p.x, p.y) for p in ps.particles])

    xs, step = np.linspace(0, 1, num=7, endpoint=False, retstep=True)
    xs += 0.5 * step

    correct_positions = set([(x, y) for x in xs for y in xs])

    assert positions == correct_positions, "sets are not the same"


def test_particles_advect():
    rp = runparams.RuntimeParameters()

    rp.params["mesh.nx"] = 8
    rp.params["mesh.ny"] = 8
    rp.params["particles.do_particles"] = 0
    rp.params["particles.n_particles"] = 50
    rp.params["particles.particle_generator"] = "grid"

    # set up sim
    sim = NullSimulation("", "", rp)

    # set up grid
    my_grid = grid_setup(rp, ng=4)
    my_data = patch.CellCenterData2d(my_grid)
    bc = bc_setup(rp)[0]
    my_data.create()
    sim.cc_data = my_data

    ps = particles.Particles(sim.cc_data, bc, rp)

    # advect with constant velocity

    # first try 0 velocity
    u = my_grid.scratch_array()
    v = my_grid.scratch_array()

    ps.update_particles(u, v, 1)

    positions = set([(p.x, p.y) for p in ps.particles.values()])

    xs, step = np.linspace(0, 1, num=7, endpoint=False, retstep=True)
    xs += 0.5 * step

    correct_positions = set([(x, y) for x in xs for y in xs])

    assert positions == correct_positions, "sets are not the same"

    # now move constant speed to the right
    u[:, :] = 1

    ps.update_particles(u, v, 1)

    positions = set([(p.x, p.y) for p in ps.particles.values()])

    correct_positions = set([(x+1, y) for x in xs for y in xs])

    assert positions == correct_positions, "sets are not the same"

    # constant speed up
    u[:, :] = 0
    v[:, :] = 1

    ps = particles.Particles(sim.cc_data, bc, rp)
    ps.update_particles(u, v, 1)

    positions = set([(p.x, p.y) for p in ps.particles.values()])

    correct_positions = set([(x, y+1) for x in xs for y in xs])

    assert positions == correct_positions, "sets are not the same"

    # constant speed right + up
    u[:, :] = 1

    ps = particles.Particles(sim.cc_data, bc, rp)
    ps.update_particles(u, v, 1)

    positions = set([(p.x, p.y) for p in ps.particles.values()])

    correct_positions = set([(x+1, y+1) for x in xs for y in xs])

    assert positions == correct_positions, "sets are not the same"
