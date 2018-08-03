from __future__ import print_function

import advection
import advection_weno.fluxes as flx
import mesh.integration as integration
import mesh.array_indexer as ai


class Simulation(advection.Simulation):

    def substep(self, myd):
        """
        take a single substep in the RK timestepping starting with the
        conservative state defined as part of myd
        """

        myg = myd.grid

        k = myg.scratch_array()

        flux_x, flux_y = flx.fluxes(myd, self.rp, self.dt)

        F_x = ai.ArrayIndexer(d=flux_x, grid=myg)
        F_y = ai.ArrayIndexer(d=flux_y, grid=myg)

        k.v()[:, :] = \
            (F_x.v() - F_x.ip(1))/myg.dx + \
            (F_y.v() - F_y.jp(1))/myg.dy

        return k

    def method_compute_timestep(self):
        """
        Compute the advective timestep (CFL) constraint.  We use the
        driver.cfl parameter to control what fraction of the CFL
        step we actually take.
        """

        cfl = self.rp.get_param("driver.cfl")

        u = self.rp.get_param("advection.u")
        v = self.rp.get_param("advection.v")

        # the timestep is 1/sum{|U|/dx}
        xtmp = max(abs(u), self.SMALL)/self.cc_data.grid.dx
        ytmp = max(abs(v), self.SMALL)/self.cc_data.grid.dy

        self.dt = cfl/(xtmp + ytmp)

    def evolve(self):
        """
        Evolve the linear advection equation through one timestep.  We only
        consider the "density" variable in the CellCenterData2d object that
        is part of the Simulation.
        """

        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()

        myd = self.cc_data

        method = self.rp.get_param("advection.temporal_method")

        rk = integration.RKIntegrator(myd.t, self.dt, method=method)
        rk.set_start(myd)

        for s in range(rk.nstages()):
            ytmp = rk.get_stage_start(s)
            ytmp.fill_BC_all()
            k = self.substep(ytmp)
            rk.store_increment(s, k)

        rk.compute_final_update()

        if self.particles is not None:
            myg = self.cc_data.grid
            u = self.rp.get_param("advection.u")
            v = self.rp.get_param("advection.v")

            u2d = myg.scratch_array() + u
            v2d = myg.scratch_array() + v

            self.particles.update_particles(self.dt, u2d, v2d)

        # increment the time
        myd.t += self.dt
        self.n += 1

        tm_evolve.end()
