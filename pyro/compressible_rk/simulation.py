import numpy as np

import pyro.compressible_rk.fluxes as flx
from pyro import compressible
from pyro.mesh import integration


class Simulation(compressible.Simulation):
    """The main simulation class for the method of lines compressible
    hydrodynamics solver"""

    def substep(self, myd):
        """
        take a single substep in the RK timestepping starting with the
        conservative state defined as part of myd
        """

        self.clean_state(myd)

        myg = myd.grid

        # source terms -- note: this dt is the entire dt, not the
        # stage's dt
        S = compressible.get_external_sources(myd.t, self.dt, myd.data,
                                              self.ivars, self.rp, myg,
                                              problem_source=self.problem_source)

        k = myg.scratch_array(nvar=self.ivars.nvar)

        flux_x, flux_y = flx.fluxes(myd, self.rp,
                                    self.ivars, self.solid, self.tc)

        for n in range(self.ivars.nvar):
            k.v(n=n)[:, :] = \
               (flux_x.v(n=n) - flux_x.ip(1, n=n))/myg.dx + \
               (flux_y.v(n=n) - flux_y.jp(1, n=n))/myg.dy + S.v(n=n)

        # finally, add the sponge source, if desired
        if self.rp.get_param("sponge.do_sponge"):
            kappa_f = compressible.get_sponge_factor(myd.data, self.ivars, self.rp, myg)

            k.v(n=self.ivars.ixmom)[:, :] -= kappa_f.v() * myd.data.v(n=self.ivars.ixmom)
            k.v(n=self.ivars.iymom)[:, :] -= kappa_f.v() * myd.data.v(n=self.ivars.iymom)

        return k

    def method_compute_timestep(self):
        """
        The timestep function computes the advective timestep (CFL)
        constraint.  The CFL constraint says that information cannot
        propagate further than one zone per timestep.

        We use the driver.cfl parameter to control what fraction of the
        CFL step we actually take.
        """

        cfl = self.rp.get_param("driver.cfl")

        # get the variables we need
        u, v, cs = self.cc_data.get_var(["velocity", "soundspeed"])

        # the timestep is min(dx/(|u| + cs), dy/(|v| + cs))
        xtmp = (abs(u) + cs)/self.cc_data.grid.dx
        ytmp = (abs(v) + cs)/self.cc_data.grid.dy

        self.dt = cfl*float(np.min(1.0/(xtmp + ytmp)))

    def evolve(self):
        """
        Evolve the equations of compressible hydrodynamics through a
        timestep dt.
        """

        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()

        myd = self.cc_data

        method = self.rp.get_param("compressible.temporal_method")

        rk = integration.RKIntegrator(myd.t, self.dt, method=method)
        rk.set_start(myd)

        for s in range(rk.nstages()):
            ytmp = rk.get_stage_start(s)
            ytmp.fill_BC_all()
            k = self.substep(ytmp)
            rk.store_increment(s, k)

        rk.compute_final_update()

        if self.particles is not None:
            self.particles.update_particles(self.dt)

        # increment the time
        myd.t += self.dt
        self.n += 1

        tm_evolve.end()
