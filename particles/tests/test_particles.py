import particles.particles as particles
import mesh.patch as patch
from util import runparams
import numpy as np
from numpy.testing import assert_array_equal
from simulation_null import NullSimulation, grid_setup, bc_setup


def test_particle():
    """
    Test Particle class
    """

    n_particles = 5

    for n in range(n_particles):
        x, y = np.random.rand(2)
        p = particles.Particle(x, y)

        assert_array_equal(p.pos(), [x, y])
        assert_array_equal(p.velocity(), [0, 0])


def setup_test(n_particles=50, extra_rp_params=None):
    """
    Function for setting up Particles tests. Would use unittest for this, but
    it doesn't seem to be possible/easy to pass in options to a setUp function.

    Sets up runtime paramters, a blank simulation, fills it with a grid and sets
    up the boundary conditions and simulation data.

    Parameters
    ----------
    n_particles : int
        Number of particles to be generated.
    extra_rp_params : dict
        Dictionary of extra rp parameters.
    """

    rp = runparams.RuntimeParameters()

    rp.params["mesh.nx"] = 8
    rp.params["mesh.ny"] = 8
    rp.params["mesh.xmin"] = 0
    rp.params["mesh.xmax"] = 1
    rp.params["mesh.ymin"] = 0
    rp.params["mesh.ymax"] = 1
    rp.params["particles.do_particles"] = 1
    n_particles = n_particles

    if extra_rp_params is not None:
        for param, value in extra_rp_params.items():
            rp.params[param] = value

    # set up sim
    sim = NullSimulation("", "", rp)

    # set up grid
    my_grid = grid_setup(rp)
    my_data = patch.CellCenterData2d(my_grid)
    bc = bc_setup(rp)[0]
    my_data.create()
    sim.cc_data = my_data

    return sim.cc_data, bc, n_particles


def test_particles_random_gen():
    """
    Test random particle generator.
    """

    myd, bc, n_particles = setup_test()

    # seed random number generator
    np.random.seed(3287469)

    ps = particles.Particles(myd, bc, n_particles, "random")
    positions = set([(p.x, p.y) for p in ps.particles.values()])

    assert len(ps.particles) == n_particles, "There should be {} particles".format(n_particles)

    # reseed random number generator
    np.random.seed(3287469)

    correct_positions = np.random.rand(n_particles, 2)
    correct_positions = set([(x, y) for (x, y) in correct_positions])

    assert positions == correct_positions, "sets are not the same"


def test_particles_grid_gen():
    """
    Test Particles grid generator.
    """

    myd, bc, n_particles = setup_test()

    ps = particles.Particles(myd, bc, n_particles, "grid")

    assert len(ps.particles) == 49, "There should be 49 particles"

    positions = set([(p.x, p.y) for p in ps.particles.values()])

    xs, step = np.linspace(0, 1, num=7, endpoint=False, retstep=True)
    xs += 0.5 * step

    correct_positions = set([(x, y) for x in xs for y in xs])

    assert positions == correct_positions, "sets are not the same"


def test_particles_array_gen():
    """
    Test Particles particle generator from input array.
    """

    myd, bc, n_particles = setup_test()

    # generate random array of particles
    init_positions = np.random.rand(n_particles, 2)

    ps = particles.Particles(myd, bc, n_particles,
                             "array", pos_array=init_positions)

    positions = set([(p.x, p.y) for p in ps.particles.values()])

    correct_positions = set([(x, y) for (x, y) in init_positions])

    assert positions == correct_positions, "sets are not the same"


def test_particles_advect():
    """
    Test particles are advected correctly and check periodic boundary conditions.
    """

    extra_rp_params = {"mesh.xlboundary": "periodic",
                       "mesh.xrboundary": "periodic",
                       "mesh.ylboundary": "periodic",
                       "mesh.yrboundary": "periodic"}

    myd, bc, n_particles = setup_test(extra_rp_params=extra_rp_params)

    ps = particles.Particles(myd, bc, n_particles, "grid")

    # advect with constant velocity

    # first try 0 velocity
    u = myd.grid.scratch_array()
    v = myd.grid.scratch_array()

    ps.update_particles(1, u, v)

    positions = set([(p.x, p.y) for p in ps.particles.values()])

    xs, step = np.linspace(0, 1, num=7, endpoint=False, retstep=True)
    xs += 0.5 * step

    correct_positions = set([(x, y) for x in xs for y in xs])

    assert positions == correct_positions, "sets are not the same"

    # now move constant speed to the right
    u[:, :] = 1

    ps.update_particles(0.1, u, v)

    positions = set([(p.x, p.y) for p in ps.particles.values()])

    correct_positions = set([((x+0.1) % 1, y) for x in xs for y in xs])

    assert positions == correct_positions, "sets are not the same"

    # constant speed up
    u[:, :] = 0
    v[:, :] = 1

    ps = particles.Particles(myd, bc, n_particles, "grid")
    ps.update_particles(0.1, u, v)

    positions = set([(p.x, p.y) for p in ps.particles.values()])

    correct_positions = set([(x, (y+0.1) % 1) for x in xs for y in xs])

    assert positions == correct_positions, "sets are not the same"

    # constant speed right + up
    u[:, :] = 1

    ps = particles.Particles(myd, bc, n_particles, "grid")
    ps.update_particles(0.1, u, v)

    positions = set([(p.x, p.y) for p in ps.particles.values()])

    correct_positions = set([((x+0.1) % 1, (y+0.1) % 1) for x in xs for y in xs])

    assert positions == correct_positions, "sets are not the same"


def test_reflect_bcs():
    """
    Test reflective boundary conditions.
    """

    extra_rp_params = {"mesh.xlboundary": "reflect-even",
                       "mesh.xrboundary": "reflect-even",
                       "mesh.ylboundary": "reflect-even",
                       "mesh.yrboundary": "reflect-even"}

    myd, bc, _ = setup_test(extra_rp_params=extra_rp_params)

    # create an array of particles at the edge of the domain.
    init_particle_positions = [[0.5, 0.03], [0.5, 0.96],
                               [0.04, 0.5], [0.97, 0.5]]

    ps = particles.Particles(myd, bc, 4, "array", init_particle_positions)

    u = myd.grid.scratch_array()
    v = myd.grid.scratch_array()

    x = myd.grid.x2d
    y = myd.grid.y2d

    # setup up velocity so that each of the particles bounces off the walls.
    u[(x < y) & (x < (1-y))] = -1
    v[(x < y) & (x > (1-y))] = 1
    u[(x > y) & (x > (1-y))] = 1
    v[(x > y) & (x < (1-y))] = -1

    ps.update_particles(0.1, u, v)

    # extract positions by their initial positions so they're in the right order
    # (as particles are stored in a dictionary they may not be returned in the
    # same order as we first inserted them.)
    positions = [ps.particles[(x, y)].pos() for (x, y) in init_particle_positions]

    correct_positions = [[0.5, 0.07], [0.5, 0.94],
                         [0.06, 0.5], [0.93, 0.5]]

    np.testing.assert_array_almost_equal(positions, correct_positions)


def test_outflow_bcs():
    """
    Test particles correctly disappear when flow outside of boundaries with
    outflow boundary conditions.
    """

    extra_rp_params = {"mesh.xlboundary": "outflow",
                       "mesh.xrboundary": "outflow",
                       "mesh.ylboundary": "outflow",
                       "mesh.yrboundary": "outflow"}

    myd, bc, _ = setup_test(extra_rp_params=extra_rp_params)

    # create an array of particles with some at the edge of the domain.
    init_particle_positions = [[0.5, 0.03], [0.5, 0.96],
                               [0.04, 0.5], [0.97, 0.5],
                               [0.5, 0.2], [0.8, 0.5]]

    ps = particles.Particles(myd, bc, 4, "array", init_particle_positions)

    u = myd.grid.scratch_array()
    v = myd.grid.scratch_array()

    x = myd.grid.x2d
    y = myd.grid.y2d

    # setup up velocity so that some of the particles are lost.
    u[(x < y) & (x < (1-y))] = -1
    v[(x < y) & (x > (1-y))] = 1
    u[(x > y) & (x > (1-y))] = 1
    v[(x > y) & (x < (1-y))] = -1

    ps.update_particles(0.1, u, v)

    assert len(ps.particles) == 2, "All but two of the particles should have flowed out of the domain"

    # extract positions by their initial positions so they're in the right order
    # (as particles are stored in a dictionary they may not be returned in the
    # same order as we first inserted them.)
    positions = [ps.particles[(x, y)].pos() for (x, y) in init_particle_positions[4:]]

    correct_positions = [[0.5, 0.1], [0.9, 0.5]]

    np.testing.assert_array_almost_equal(positions, correct_positions)
