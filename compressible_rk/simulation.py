from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import compressible.eos as eos
import mesh.patch as patch
import compressible
import compressible_rk.fluxes as flx
from util import profile


class Simulation(compressible.Simulation):

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
        ener = myd.get_var("energy")

        ymom_src = myg.scratch_array()
        ymom_src.v()[:,:] = dens.v()[:,:]*grav

        E_src = myg.scratch_array()
        E_src.v()[:,:] = ymom.v()[:,:]*grav

        k = myg.scratch_array(nvar=self.vars.nvar)

        flux_x, flux_y = flx.fluxes(myd, self.rp,
                                    self.vars, self.solid, self.tc)

        for n in range(self.vars.nvar):
            k.v(n=n)[:,:] = \
               (flux_x.v(n=n) - flux_x.ip(1, n=n))/myg.dx + \
               (flux_y.v(n=n) - flux_y.jp(1, n=n))/myg.dy

        k.v(n=self.vars.iymom)[:,:] += ymom_src.v()[:,:]
        k.v(n=self.vars.iener)[:,:] += E_src.v()[:,:]

        return k


    def evolve(self):
        """
        Evolve the equations of compressible hydrodynamics through a
        timestep dt.
        """

        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()


        myg = self.cc_data.grid

        myd = self.cc_data

        order = self.rp.get_param("compressible.temporal_order")

        if order == 2:
            # time-integration -- RK2
            myd_nhalf = patch.cell_center_data_clone(myd)

            # initial slopes and n+1/2 state
            k1 = self.substep(myd)
            for n in range(self.vars.nvar):
                var = myd_nhalf.get_var_by_index(n)
                var.v()[:,:] += 0.5*self.dt*k1.v(n=n)[:,:]

            myd_nhalf.fill_BC_all()

            # updated slopes, starting with the n+1/2 state
            k2 = self.substep(myd_nhalf)

            # final update
            for n in range(self.vars.nvar):
                var = myd.get_var_by_index(n)
                var.v()[:,:] += self.dt*k2.v(n=n)[:,:]

        elif order == 4:

            # time-integration -- RK4

            # first slope (k1) is f(U^n)
            k1 = self.substep(myd)

            # second slope (k2) is f(U^n + 0.5*dt*k1)
            myd1 = patch.cell_center_data_clone(myd)
            for n in range(self.vars.nvar):
                var = myd1.get_var_by_index(n)
                var.v()[:,:] += 0.5*self.dt*k1.v(n=n)[:,:]

            myd1.fill_BC_all()
            k2 = self.substep(myd1)

            # third slope (k3) is f(U^n + 0.5*dt*k2)
            myd2 = patch.cell_center_data_clone(myd)
            for n in range(self.vars.nvar):
                var = myd2.get_var_by_index(n)
                var.v()[:,:] += 0.5*self.dt*k2.v(n=n)[:,:]

            myd2.fill_BC_all()
            k3 = self.substep(myd2)

            # last slope (k4) is f(U^n + dt*k3)
            myd3 = patch.cell_center_data_clone(myd)
            for n in range(self.vars.nvar):
                var = myd3.get_var_by_index(n)
                var.v()[:,:] += self.dt*k3.v(n=n)[:,:]

            myd3.fill_BC_all()
            k4 = self.substep(myd3)

            # final update
            for n in range(self.vars.nvar):
                var = myd.get_var_by_index(n)
                var.v()[:,:] += (self.dt/6.0)*(k1.v(n=n)[:,:] + 2.0*k2.v(n=n)[:,:] + 
                                               2.0*k3.v(n=n)[:,:] + k4.v(n=n)[:,:])


        # increment the time
        myd.t += self.dt
        self.n += 1

        tm_evolve.end()
