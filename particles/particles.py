"""
Stores and manages particles and updates their positions based
on the velocity on the grid.
"""

import numpy as np
from util import msg


class Particle(object):
    """
    Class to hold properties of a single (massless) particle.

    This class could be extended (i.e. inherited from) to
    model e.g. massive/charged particles.
    """

    def __init__(self, x, y, u=0, v=0):
        self.x = x
        self.y = y
        self.u = u
        self.v = v

    def pos(self):
        """
        Return position vector.
        """
        return np.array([self.x, self.y])

    def velocity(self):
        """
        Return velocity vector.
        """
        return np.array([self.u, self.v])

    def update(self, u, v, dt):
        """
        Advect the particle and update its velocity.
        """
        self.u = u
        self.v = v
        self.x += u * dt
        self.y += v * dt

    def interpolate_velocity(self, myg, u, v):
        """
        Interpolate the x- and y-velocities defined on grid myg to the
        particle's position.

        Parameters
        ----------
        myg : Grid2d
            grid which the velocities are defined on
        u : ArrayIndexer
            x-velocity
        v : ArrayIndexer
            y_velocity
        """

        # find what cell it lives in
        x_idx = (self.x - myg.xmin) / myg.dx - 0.5
        y_idx = (self.y - myg.ymin) / myg.dy - 0.5

        x_frac = x_idx % 1
        y_frac = y_idx % 1

        # get the index of the particle's closest bottom left cell
        # we'll add one as going to use buf'd grids -
        # this will catch the cases where the particle is on the edges
        # of the grid.
        x_idx = int(x_idx) + 1
        y_idx = int(y_idx) + 1

        # interpolate velocity
        u_vel = (1-x_frac)*(1-y_frac)*u.v(buf=1)[x_idx, y_idx] + \
                x_frac*(1-y_frac)*u.v(buf=1)[x_idx+1, y_idx] + \
                (1-x_frac)*y_frac*u.v(buf=1)[x_idx, y_idx+1] + \
                x_frac*y_frac*u.v(buf=1)[x_idx+1, y_idx+1]

        v_vel = (1-x_frac)*(1-y_frac)*v.v(buf=1)[x_idx, y_idx] + \
                x_frac*(1-y_frac)*v.v(buf=1)[x_idx+1, y_idx] + \
                (1-x_frac)*y_frac*v.v(buf=1)[x_idx, y_idx+1] + \
                x_frac*y_frac*v.v(buf=1)[x_idx+1, y_idx+1]

        return u_vel, v_vel


class Particles(object):
    """
    Class to hold multiple particles.
    """

    def __init__(self, sim_data, bc, n_particles, particle_generator="grid",
                 pos_array=None, init_array=None):
        """
        Initialize the Particles object.

        Particles are stored as a dictionary, with their keys being tuples
        of their initial position. This was done in order to have a simple way
        to access the initial particle positions when plotting.

        However, this assumes that no two particles are
        initialised with the same initial position, which is fine for the
        massless particle case, however could no longer be a sensible thing
        to do if have particles have other properties (e.g. mass).

        Parameters
        ----------
        sim_data : CellCenterData2d object
            The cell-centered simulation data
        bc : BC object
            Boundary conditions
        n_particles : int
            Number of particles
        particle_generator : string or function
            String with generator name of custom particle generator function
        pos_array : float array
            Array of particle positions to use with particle initialization
        init_array : float array
            Array of initial particle positions required for plotting from file.
        """

        self.sim_data = sim_data
        self.bc = bc
        self.particles = dict()

        if n_particles <= 0:
            msg.fail("ERROR: n_particles = %s <= 0" % (n_particles))

        if callable(particle_generator):  # custom particle generator function
            self.particles = particle_generator(n_particles)
        else:
            if particle_generator == "random":
                self.randomly_generate_particles(n_particles)
            elif particle_generator == "grid":
                self.grid_generate_particles(n_particles)
            elif particle_generator == "array":
                self.array_generate_particles(pos_array, init_array)
            else:
                msg.fail("ERROR: do not recognise particle generator %s"
                         % (particle_generator))

        self.n_particles = len(self.particles)

    def randomly_generate_particles(self, n_particles):
        """
        Randomly generate n_particles.
        """
        myg = self.sim_data.grid

        positions = np.random.rand(n_particles, 2)

        positions[:, 0] = positions[:, 0] * (myg.xmax - myg.xmin) + \
            myg.xmin

        positions[:, 1] = positions[:, 1] * (myg.ymax - myg.ymin) + \
            myg.ymin

        for (x, y) in positions:
            self.particles[(x, y)] = Particle(x, y)

    def grid_generate_particles(self, n_particles):
        """
        Generate particles equally spaced across the grid.
        Currently has the same number of particles in the x and y
        directions (so dx != dy unless the domain is square) -
        may be better to scale this.

        If necessary, shall increase/decrease n_particles
        in order to fill grid.
        """

        sq_n_particles = int(round(np.sqrt(n_particles)))

        if sq_n_particles**2 != n_particles:
            msg.warning("WARNING: Changing number of particles from {} to {}".format(n_particles, sq_n_particles**2))

        myg = self.sim_data.grid

        xs, step = np.linspace(myg.xmin, myg.xmax, num=sq_n_particles, endpoint=False, retstep=True)
        xs += 0.5 * step
        ys, step = np.linspace(myg.ymin, myg.ymax, num=sq_n_particles, endpoint=False, retstep=True)
        ys += 0.5 * step
        for x in xs:
            for y in ys:
                self.particles[(x, y)] = Particle(x, y)

    def array_generate_particles(self, pos_array, init_array=None):
        """
        Generate particles based on array of their positions. This is used
        when reading particle data from file.

        Parameters
        ----------
        pos_array : float array
            Array of particle positions.
        init_array : float array
            Array of initial particle positions.
        """
        # check a pos_array has been passed to the constructor
        if pos_array is None:
            msg.fail("ERROR: Array of particle positions has not been passed into Particles constructor.\
            Cannot generate particles.")

        if init_array is None:
            for (x, y) in pos_array:
                self.particles[(x, y)] = Particle(x, y)
        else:
            for (ix, iy), (x, y) in zip(init_array, pos_array):
                self.particles[(ix, iy)] = Particle(x, y)

    def update_particles(self, dt, u=None, v=None):
        """
        Update the particles on the grid. This is based off the
        AdvectWithUcc function in AMReX, which used the midpoint
        method to advance particles using the cell-centered velocity.

        We will explicitly pass in u and v if they cannot be accessed from the
        sim_data using get_var("velocity").

        Parameters
        ----------
        dt : float
            timestep
        u : ArrayIndexer object
            x-velocity
        v : ArrayIndexer object
            y-velocity
        """
        myg = self.sim_data.grid

        if (u is None) and (v is None):
            u, v = self.sim_data.get_var("velocity")
        elif u is None:
            u = self.sim_data.get_var("x-velocity")
        elif v is None:
            v = self.sim_data.get_var("y-velocity")

        for k, p in self.particles.items():

            # save old position and predict location at dt/2
            initial_position = p.pos()

            u_vel, v_vel = p.interpolate_velocity(myg, u, v)

            p.update(u_vel, v_vel, 0.5*dt)

            # find velocity at dt/2
            u_vel, v_vel = p.interpolate_velocity(myg, u, v)

            # update to final time using original position and vel at dt/2
            p.x, p.y = initial_position

            p.update(u_vel, v_vel, dt)

        self.enforce_particle_boundaries()

    def enforce_particle_boundaries(self):
        """
        Enforce the particle boundaries

        TODO: copying the dict and adding everything back again is messy
        - think of a better way to do this? Did it this way so don't have
        to remove items from a dictionary while iterating it for outflow
        boundaries.
        """
        old_particles = self.particles.copy()
        self.particles = dict()

        myg = self.sim_data.grid

        xlb = self.bc.xlb
        xrb = self.bc.xrb
        ylb = self.bc.ylb
        yrb = self.bc.yrb

        while old_particles:
            k, p = old_particles.popitem()

            # -x boundary
            if p.x < myg.xmin:
                if xlb in ["outflow", "neumann"]:
                    continue
                elif xlb == "periodic":
                    p.x = myg.xmax + p.x - myg.xmin
                elif xlb in ["reflect-even", "reflect-odd", "dirichlet"]:
                    p.x = 2 * myg.xmin - p.x
                else:
                    msg.fail("ERROR: xlb = %s invalid BC for particles" % (xlb))

            # +x boundary
            if p.x > myg.xmax:
                if xrb in ["outflow", "neumann"]:
                    continue
                elif xrb == "periodic":
                    p.x = myg.xmin + p.x - myg.xmax
                elif xrb in ["reflect-even", "reflect-odd", "dirichlet"]:
                    p.x = 2 * myg.xmax - p.x
                else:
                    msg.fail("ERROR: xrb = %s invalid BC for particles" % (xrb))

            # -y boundary
            if p.y < myg.ymin:
                if ylb in ["outflow", "neumann"]:
                    continue
                elif ylb == "periodic":
                    p.y = myg.ymax + p.y - myg.ymin
                elif ylb in ["reflect-even", "reflect-odd", "dirichlet"]:
                    p.y = 2 * myg.ymin - p.y
                else:
                    msg.fail("ERROR: ylb = %s invalid BC for particles" % (ylb))

            # +y boundary
            if p.y > myg.ymax:
                if yrb in ["outflow", "neumann"]:
                    continue
                elif yrb == "periodic":
                    p.y = myg.ymin + p.y - myg.ymax
                elif yrb in ["reflect-even", "reflect-odd", "dirichlet"]:
                    p.y = 2 * myg.ymax - p.y
                else:
                    msg.fail("ERROR: yrb = %s invalid BC for particles" % (yrb))

            self.particles[k] = p

        self.n_particles = len(self.particles)

    def get_positions(self):
        """
        Return an array of current particle positions.
        """
        return np.array([[p.x, p.y] for p in self.particles.values()])

    def get_init_positions(self):
        """
        Return initial positions of the particles as an array.

        We defined the particles as a dictionary with their initial positions
        as the keys, so this just becomes a restructuring operation.
        """
        return np.array([[x, y] for (x, y) in self.particles.keys()])

    def write_particles(self, f):
        """
        Output the particles' positions (and initial positions) to an HDF5 file.

        Parameters
        ----------
        f : h5py object
            HDF5 file to write to
        """

        gparticles = f.create_group("particles")
        gparticles.create_dataset("init_particle_positions",
            data=self.get_init_positions())
        gparticles.create_dataset("particle_positions",
            data=self.get_positions())
