from __future__ import print_function

import importlib
import numpy as np
import matplotlib.pyplot as plt

import advection
import advection_rk.fluxes as flx
import mesh.patch as patch
import mesh.array_indexer as ai

from util import profile


class Simulation(advection.Simulation):

    def substep(self, myd):
        """
        take a single substep in the RK timestepping starting with the
        conservative state defined as part of myd
        """

        myg = myd.grid

        k = myg.scratch_array()

        flux_x, flux_y =  flx.fluxes(myd, self.rp, self.dt)

        F_x = ai.ArrayIndexer(d=flux_x, grid=myg)
        F_y = ai.ArrayIndexer(d=flux_y, grid=myg)

        k.v()[:,:] = \
            (F_x.v() - F_x.ip(1))/myg.dx + \
            (F_y.v() - F_y.jp(1))/myg.dy

        return k


    def evolve(self):
        """
        Evolve the linear advection equation through one timestep.  We only
        consider the "density" variable in the CellCenterData2d object that
        is part of the Simulation.
        """

        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()

        myg = self.cc_data.grid
        myd = self.cc_data

        order = self.rp.get_param("advection.temporal_order")

        if order == 2:

            # time-integration -- RK2
            myd_nhalf = patch.cell_center_data_clone(myd)

            # initial slopes and n+1/2 state
            k1 = self.substep(myd)
            var = myd_nhalf.get_var("density")
            var.v()[:,:] += 0.5*self.dt*k1.v()[:,:]

            myd_nhalf.fill_BC_all()

            # updated slopes, starting with the n+1/2 state
            k2 = self.substep(myd_nhalf)

            # final update
            var = myd.get_var("density")
            var.v()[:,:] += self.dt*k2.v()[:,:]

        elif order == 4:

            # time-integration -- RK4

            # first slope (k1) is f(U^n)
            k1 = self.substep(myd)

            # second slope (k2) is f(U^n + 0.5*dt*k1)            
            myd1 = patch.cell_center_data_clone(myd)
            var = myd1.get_var("density")
            var.v()[:,:] += 0.5*self.dt*k1.v()[:,:]

            myd1.fill_BC_all()
            k2 = self.substep(myd1)

            # third slope (k3) is f(U^n + 0.5*dt*k2)
            myd2 = patch.cell_center_data_clone(myd)
            var = myd2.get_var("density")
            var.v()[:,:] += 0.5*self.dt*k2.v()[:,:]

            myd2.fill_BC_all()
            k3 = self.substep(myd2)

            # last slope (k4) is f(U^n + dt*k3)
            myd3 = patch.cell_center_data_clone(myd)
            var = myd3.get_var("density")
            var.v()[:,:] += self.dt*k3.v()[:,:]

            myd3.fill_BC_all()
            k4 = self.substep(myd3)

            # final update
            var = myd.get_var("density")
            var.v()[:,:] += (self.dt/6.0)*(k1.v()[:,:] + 2.0*k2.v()[:,:] + 2.0*k3.v()[:,:] + k4.v()[:,:])

        elif order == 103:

            # time-integration -- SSP RK3 (from Shu & Osher 1989)

            # compute u^1 = u^n + dt * L(u^n)
            k0 = self.substep(myd)
            u1 = patch.cell_center_data_clone(myd)
            var1 = u1.get_var("density")
            var1.v()[:,:] += self.dt*k0.v()[:,:]

            u1.fill_BC_all()

            # compute u^2 = (3/4) u^n + (1/4) u^1 + (1/4) dt * L(u^1)
            k1 = self.substep(u1)
            u2 = patch.cell_center_data_clone(myd)
            varn = myd.get_var("density")
            var1 = u1.get_var("density")
            var2 = u2.get_var("density")
            var2.v()[:,:] = 0.75*varn.v()[:,:] + 0.25*var1.v()[:,:] + 0.25*self.dt*k1.v()[:,:]

            u2.fill_BC_all()

            # compute u^new = (1/3) u^n + (2/3) u^2 + (2/3) dt L(u^2)
            k2 = self.substep(u2)
            var = myd.get_var("density")
            var2 = u2.get_var("density")
            var.v()[:,:] = (1.0/3.0)*var.v()[:,:] + (2.0/3.0)*var2.v()[:,:] + (2.0/3.0)*self.dt*k2.v()[:,:]



        # increment the time
        myd.t += self.dt
        self.n += 1

        tm_evolve.end()

