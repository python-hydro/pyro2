from __future__ import print_function

import numpy as np

import mesh.integration as integration
import compressible
import compressible_rk.fluxes as flx


class Simulation(compressible.Simulation):
    """The main simulation class for the method of lines compressible
    hydrodynamics solver"""

    def substep(self, myd):
        """
        take a single substep in the RK timestepping starting with the
        conservative state defined as part of myd
        """

        myg = myd.grid
        grav = self.rp.get_param("compressible.grav")

        # compute the source terms
        dens = myd.get_var("density")
        ymom = myd.get_var("y-momentum")

        ymom_src = myg.scratch_array()
        ymom_src.v()[:, :] = dens.v()[:, :]*grav

        E_src = myg.scratch_array()
        E_src.v()[:, :] = ymom.v()[:, :]*grav

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
