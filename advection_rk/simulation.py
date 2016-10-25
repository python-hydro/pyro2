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

        order = 4

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
            myd1 = patch.cell_center_data_clone(myd)
            myd2 = patch.cell_center_data_clone(myd)
            myd3 = patch.cell_center_data_clone(myd)

            k1 = self.substep(myd)
            var = myd1.get_var("density")
            var.v()[:,:] += 0.5*self.dt*k1.v()[:,:]

            myd1.fill_BC_all()

            k2 = self.substep(myd1)
            var = myd2.get_var("density")
            var.v()[:,:] += 0.5*self.dt*k2.v()[:,:]

            myd2.fill_BC_all()

            k3 = self.substep(myd2)
            var = myd3.get_var("density")
            var.v()[:,:] += self.dt*k3.v()[:,:]

            myd3.fill_BC_all()

            # updated slopes, starting with the n+1/2 state
            k4 = self.substep(myd3)

            # final update
            var = myd.get_var("density")
            var.v()[:,:] += (self.dt/6.0)*(k1.v()[:,:] + 2.0*k2.v()[:,:] + 2.0*k3.v()[:,:] + k4.v()[:,:])


        # increment the time
        myd.t += self.dt
        self.n += 1

        tm_evolve.end()

