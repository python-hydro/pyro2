from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from compressible.problems import *
import compressible.eos as eos
import mesh.patch as patch
import compressible
from compressible_rk.fluxes import *
from util import profile


class Simulation(compressible.Simulation):

    def substep(self, myd):
        """
        U is a numpy array, [i,j,n], where n is variable
        for the conserved quantities
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

        flux_x, flux_y = fluxes(myd, self.rp,
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
        myd_nhalf = patch.cell_center_data_clone(myd)

        # time-integration -- RK2

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

        # increment the time
        myd.t += self.dt
        self.n += 1

        tm_evolve.end()
