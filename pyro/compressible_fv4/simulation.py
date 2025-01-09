import pyro.compressible_fv4.fluxes as flx
from pyro import compressible_rk
from pyro.compressible import get_external_sources, get_sponge_factor
from pyro.mesh import fv, integration


class Simulation(compressible_rk.Simulation):

    def __init__(self, solver_name, problem_name, problem_func, rp, *,
                 problem_finalize_func=None, problem_source_func=None,
                 timers=None, data_class=fv.FV2d):
        super().__init__(solver_name, problem_name, problem_func, rp,
                         problem_finalize_func=problem_finalize_func,
                         problem_source_func=problem_source_func,
                         timers=timers, data_class=data_class)

    def substep(self, myd):
        """
        compute the advective source term for the given state
        """

        myg = myd.grid

        # compute the source terms -- we need to do this first
        # using the cell-center data and then convert it back to
        # averages

        U_cc = myg.scratch_array(nvar=self.ivars.nvar)
        for n in range(self.ivars.nvar):
            U_cc[:, :, n] = myd.to_centers(myd.names[n])

        # cell-centered sources
        S = get_external_sources(myd.t, self.dt, U_cc,
                                 self.ivars, self.rp, myg,
                                 problem_source=self.problem_source)

        # bring the sources back to averages -- we only care about
        # the interior (no ghost cells)
        for n in range(self.ivars.nvar):
            S.v(n=n)[:, :] -= myg.dx**2 * S.lap(n=n) / 24.0

        k = myg.scratch_array(nvar=self.ivars.nvar)

        flux_x, flux_y = flx.fluxes(myd, self.rp, self.ivars)

        for n in range(self.ivars.nvar):
            k.v(n=n)[:, :] = \
               (flux_x.v(n=n) - flux_x.ip(1, n=n))/myg.dx + \
               (flux_y.v(n=n) - flux_y.jp(1, n=n))/myg.dy + S.v(n=n)

        # finally, add the sponge source, if desired
        if self.rp.get_param("sponge.do_sponge"):
            kappa_f = get_sponge_factor(myd.data, self.ivars, self.rp, myg)

            k.v(n=self.ivars.ixmom)[:, :] -= kappa_f.v() * myd.data.v(n=self.ivars.ixmom)
            k.v(n=self.ivars.iymom)[:, :] -= kappa_f.v() * myd.data.v(n=self.ivars.iymom)

        return k

    def preevolve(self):
        """Since we are 4th order accurate we need to make sure that we
        initialized with accurate zone-averages, so the preevolve for
        this solver assumes that the initialization was done to
        cell-centers and converts it to cell-averages."""

        # we just initialized cell-centers, but we need to store averages
        for var in self.cc_data.names:
            self.cc_data.from_centers(var)

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
