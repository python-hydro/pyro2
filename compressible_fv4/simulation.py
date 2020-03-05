from __future__ import print_function

import mesh.integration as integration
import mesh.fv as fv
import compressible_rk
import compressible_fv4.fluxes as flx


class Simulation(compressible_rk.Simulation):

    def __init__(self, solver_name, problem_name, rp, timers=None, data_class=fv.FV2d):
        super().__init__(solver_name, problem_name, rp, timers=timers, data_class=data_class)

    def initialize(self, ng=5):
        super().initialize(ng=ng)

    def substep(self, myd):
        """
        compute the advective source term for the given state
        """

        myg = myd.grid
        grav = self.rp.get_param("compressible.grav")

        # compute the source terms -- we need to do these to 4th
        # order.  Start by evaluating the sources using the
        # cell-center quantities (including one ghost cell.
        dens_cc = myd.to_centers("density")
        ymom_cc = myd.to_centers("y-momentum")

        ymom_src = myg.scratch_array()
        ymom_src.v(buf=1)[:, :] = dens_cc.v(buf=1)[:, :]*grav

        E_src = myg.scratch_array()
        E_src.v(buf=1)[:, :] = ymom_cc.v(buf=1)[:, :]*grav

        # now bring back to averages -- we only need this in the
        # interior (no ghost cells)
        ymom_src.v()[:, :] = ymom_src.v()[:, :] - myg.dx**2*ymom_src.lap()/24.0
        E_src.v()[:, :] = E_src.v()[:, :] - myg.dx**2*E_src.lap()/24.0

        k = myg.scratch_array(nvar=self.ivars.nvar)

        flux_x, flux_y = flx.fluxes(myd, self.rp,
                                    self.ivars, self.solid, self.tc)

        for n in range(self.ivars.nvar):
            k.v(n=n)[:, :] = \
               (flux_x.v(n=n) - flux_x.ip(1, n=n))/myg.dx + \
               (flux_y.v(n=n) - flux_y.jp(1, n=n))/myg.dy

        k.v(n=self.ivars.iymom)[:, :] += ymom_src.v()[:, :]
        k.v(n=self.ivars.iener)[:, :] += E_src.v()[:, :]

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
