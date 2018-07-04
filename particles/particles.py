"""
Stores and manages particles and updates their positions based
on the velocity on the grid.
"""

import numpy as np
import mesh.reconstruction as reconstruction
from util import msg


class Particle(object):
    """
    Class to hold properties of a single particle.

    Not sure need velocity (or mass?), but will store it
    here for now.
    """

    def __init__(self, x, y, u=0, v=0, mass=1):
        self.x = x
        self.y = y
        self.u = u
        self.v = v
        self.mass = mass

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

    def advect(self, u, v, dt):
        """
        Advect the particle and update its velocity.
        """
        self.u = u
        self.v = v
        self.x += u * dt
        self.y += v * dt


class Particles(object):

    def __init__(self, sim_data, bc, rp):
        """
        Initialize the Particles object.

        Parameters
        ----------
        sim_data : CellCenterData2d object
            The simulation data
        """

        self.sim_data = sim_data
        self.bc = bc
        self.particles = dict()
        self.rp = rp

        # TODO: read something from rp here to determine how to
        # generate the particles - for now, we shall assume random.

        particle_generator = self.rp.get_param("particles.particle_generator")
        n_particles = self.rp.get_param("particles.n_particles")
        if n_particles <= 0:
            msg.fail("ERROR: n_particles = %s <= 0" % (n_particles))

        if particle_generator == "random":
            self.randomly_generate_particles(n_particles)
        elif particle_generator == "grid":
            self.grid_generate_particles(n_particles)
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

        If necessary, shall increase/decrease n_particles
        in order to
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

    def update_particles(self, u, v, dt, limiter=0):
        """
        Update the particles on the grid. To do this, we need to
        calculate the velocity at the particle's position (do we
        do this by interpolating - ie assuming the grid velocities
        to live at points - or by using the velocity in the cell well
        the particle is - ie assuming the grid velocities to live in the
        entire cell. I think I'll try using the same projection methods
        used in other codes?)

        We will explicitly pass in u and v here as these are accessed
        differently in different problems.
        """
        myg = self.sim_data.grid
        # myd = self.sim_data.data

        # limit the velocity

        ldelta_ux = reconstruction.limit(u, myg, 1, limiter)
        ldelta_uy = reconstruction.limit(u, myg, 2, limiter)

        ldelta_vx = reconstruction.limit(v, myg, 1, limiter)
        ldelta_vy = reconstruction.limit(v, myg, 2, limiter)

        for k, p in self.particles.items():
            # find what cell it lives in
            x_idx = (p.x - myg.xmin) / myg.dx - 0.5
            y_idx = (p.y - myg.ymin) / myg.dy - 0.5

            x_frac = x_idx % 1
            y_frac = y_idx % 1

            x_idx = int(round(x_idx))
            y_idx = int(round(y_idx))

            if x_frac > 0.5 and x_idx+1 < myg.nx:
                x_frac -= 1
                x_idx += 1
            if y_frac > 0.5 and y_idx+1 < myg.ny:
                y_frac -= 1
                y_idx += 1

            if x_idx >= myg.nx:
                x_frac += (x_idx - myg.nx) + 1
                x_idx = myg.nx - 1
            if y_idx >= myg.ny:
                y_frac += (y_idx - myg.ny) + 1
                y_idx = myg.ny - 1

            u_vel = u.v()[x_idx, y_idx]
            v_vel = v.v()[x_idx, y_idx]
            cx = u_vel * dt / myg.dx
            cy = v_vel * dt / myg.dy

            # normal velocity
            if (u_vel*x_frac) < 0:
                u_vel -= x_frac*(1.0 + cx)*ldelta_ux.v()[x_idx, y_idx]
            else:
                u_vel += x_frac*(1.0 - cx)*ldelta_ux.v()[x_idx, y_idx]

            if (v_vel*y_frac) < 0:
                v_vel -= y_frac*(1.0 + cy)*ldelta_vy.v()[x_idx, y_idx]
            else:
                v_vel += y_frac*(1.0 - cy)*ldelta_vy.v()[x_idx, y_idx]
            #
            # # transverse velocity
            u_vel += y_frac * ldelta_uy.v()[x_idx, y_idx]
            v_vel += x_frac * ldelta_vx.v()[x_idx, y_idx]

            p.advect(u_vel, v_vel, dt)

    def enforce_particle_boundaries(self):
        """
        Enforce the particle boundaries

        TODO: copying the set and adding everything back again is messy
        - think of a better way to do this?
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
                elif xlb in ["reflect-even", "reflect-odd"]:
                    p.x = 2 * myg.xmin - p.x
                else:
                    msg.fail("ERROR: xlb = %s invalid BC" % (xlb))

            # +x boundary
            if p.x > myg.xmax:
                if xrb in ["outflow", "neumann"]:
                    continue
                elif xrb == "periodic":
                    p.x = myg.xmin + p.x - myg.xmax
                elif xrb in ["reflect-even", "reflect-odd"]:
                    p.x = 2 * myg.xmax - p.x
                else:
                    msg.fail("ERROR: xrb = %s invalid BC" % (xrb))

            # -y boundary
            if p.y < myg.ymin:
                if ylb in ["outflow", "neumann"]:
                    continue
                elif ylb == "periodic":
                    p.y = myg.ymax + p.y - myg.ymin
                elif ylb in ["reflect-even", "reflect-odd"]:
                    p.y = 2 * myg.ymin - p.y
                else:
                    msg.fail("ERROR: ylb = %s invalid BC" % (ylb))

            # +y boundary
            if p.y > myg.ymax:
                if yrb in ["outflow", "neumann"]:
                    continue
                elif yrb == "periodic":
                    p.y = myg.ymin + p.y - myg.ymax
                elif yrb in ["reflect-even", "reflect-odd"]:
                    p.y = 2 * myg.ymax - p.y
                else:
                    msg.fail("ERROR: yrb = %s invalid BC" % (yrb))

            self.particles[k] = p

        self.n_particles = len(self.particles)

    def get_positions(self):
        """
        Return an array of particle positions.
        """
        return np.array([[p.x, p.y] for p in self.particles.values()])

    def get_init_positions(self):
        """
        We defined the particles as a dictionary with their initial positions
        as the keys, so this just becomes a restructuring operation.
        """
        return np.array([[x, y] for (x,y) in self.particles.keys()])

    def write_particles(self, filename):
        """
        Output the particles to an HDF5 file
        """

        pass
