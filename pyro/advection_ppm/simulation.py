import pyro.advection.advective_fluxes as flx
from pyro import advection
from pyro.advection_ppm.interface import ppm_interface


class Simulation(advection.Simulation):

    def evolve(self):
        """
        Evolve the linear advection equation through one timestep.  We only
        consider the "density" variable in the CellCenterData2d object that
        is part of the Simulation.
        """

        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()

        dtdx = self.dt/self.cc_data.grid.dx
        dtdy = self.dt/self.cc_data.grid.dy

        Fx, Fy = flx.unsplit_fluxes(self.cc_data, self.rp, self.dt, "density", ppm_interface)

        dens = self.cc_data.get_var("density")

        dens.v()[:, :] = dens.v() + dtdx*(Fx.v() - Fx.ip(1)) + \
                                    dtdy*(Fy.v() - Fy.jp(1))

        if self.particles is not None:
            myg = self.cc_data.grid
            u = self.rp.get_param("advection.u")
            v = self.rp.get_param("advection.v")

            u2d = myg.scratch_array() + u
            v2d = myg.scratch_array() + v

            self.particles.update_particles(self.dt, u2d, v2d)

        # increment the time
        self.cc_data.t += self.dt
        self.n += 1

        tm_evolve.end()
