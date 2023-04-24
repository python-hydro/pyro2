from pyro.burgers import Simulation as burgers_sim
from pyro.burgers import burgers_interface
from pyro.mesh import reconstruction
from pyro.viscous_burgers import interface


class Simulation(burgers_sim):

    def evolve(self):
        """
        Evolve the viscous burgers equation through one timestep.
        """

        myg = self.cc_data.grid

        u = self.cc_data.get_var("x-velocity")
        v = self.cc_data.get_var("y-velocity")

        # --------------------------------------------------------------------------
        # monotonized central differences
        # --------------------------------------------------------------------------

        limiter = self.rp.get_param("advection.limiter")
        eps = self.rp.get_param("diffusion.eps")

        # Give da/dx and da/dy using input: (state, grid, direction, limiter)

        ldelta_ux = reconstruction.limit(u, myg, 1, limiter)
        ldelta_uy = reconstruction.limit(u, myg, 2, limiter)
        ldelta_vx = reconstruction.limit(v, myg, 1, limiter)
        ldelta_vy = reconstruction.limit(v, myg, 2, limiter)

        # Compute the advective fluxes
        # Get u, v fluxes

        # Get the interface states without transverse or diffusion corrections
        u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr = burgers_interface.get_interface_states(myg, self.dt,
                                                                                                u, v,
                                                                                                ldelta_ux, ldelta_vx,
                                                                                                ldelta_uy, ldelta_vy)

        # Apply diffusion correction terms to the interface states
        u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr = interface.apply_diffusion_corrections(myg, self.dt, eps,
                                                                                                u, v,
                                                                                                u_xl, u_xr,
                                                                                                u_yl, u_yr,
                                                                                                v_xl, v_xr,
                                                                                                v_yl, v_yr)

        # Apply transverse correction terms to the interface states
        u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr = burgers_interface.apply_transverse_corrections(myg, self.dt,
                                                                                                u_xl, u_xr,
                                                                                                u_yl, u_yr,
                                                                                                v_xl, v_xr,
                                                                                                v_yl, v_yr)

        # Construct the interface fluxes
        u_flux_x, u_flux_y, v_flux_x, v_flux_y = burgers_interface.construct_unsplit_fluxes(myg,
                                                                                            u_xl, u_xr,
                                                                                            u_yl, u_yr,
                                                                                            v_xl, v_xr,
                                                                                            v_yl, v_yr)

        # Compute the advective source terms for diffusion using flux computed above

        A_u = myg.scratch_array()
        A_v = myg.scratch_array()

        A_u.v()[:, :] = (u_flux_x.ip(1) - u_flux_x.v()) / myg.dx + \
                        (u_flux_y.jp(1) - u_flux_y.v()) / myg.dy

        A_v.v()[:, :] = (v_flux_x.ip(1) - v_flux_x.v()) / myg.dx + \
                        (v_flux_y.jp(1) - v_flux_y.v()) / myg.dy

        # Update state by doing diffusion update with extra advective source term.

        interface.diffuse(self.cc_data, self.rp, self.dt, "x-velocity", A_u)
        interface.diffuse(self.cc_data, self.rp, self.dt, "y-velocity", A_v)

        if self.particles is not None:

            u2d = self.cc_data.get_var("x-velocity")
            v2d = self.cc_data.get_var("y-velocity")

            self.particles.update_particles(self.dt, u2d, v2d)

        # increment the time
        self.cc_data.t += self.dt
        self.n += 1
