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

    def evolve(self):
        """
        Evolve the equations of compressible hydrodynamics through a
        timestep dt.
        """

        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()


        grav = self.rp.get_param("compressible.grav")

        myg = self.cc_data.grid

        cc_data_old = patch.cell_center_data_clone(self.cc_data)

        dtdx = self.dt/myg.dx
        dtdy = self.dt/myg.dy


        # time-integration loop -- RK2
        for istep in range(2):
            myd = self.cc_data

            dens = myd.get_var("density")
            ymom = myd.get_var("y-momentum")
            ener = myd.get_var("energy")

            old_dens = dens.copy()
            old_ymom = ymom.copy()

            flux_x, flux_y = fluxes(myd, self.rp, 
                                    self.vars, self.solid, self.tc, self.dt)

            # increment by dt/2 for i = 0, dt for i = 1
            if istep == 0:
                for n in range(self.vars.nvar):
                    var = myd.get_var_by_index(n)

                    var.v()[:,:] += \
                        0.5*dtdx*(flux_x.v(n=n) - flux_x.ip(1, n=n)) + \
                        0.5*dtdy*(flux_y.v(n=n) - flux_y.jp(1, n=n))
            else:
                for n in range(self.vars.nvar):
                    var = cc_data_old.get_var_by_index(n)

                    var.v()[:,:] += \
                        dtdx*(flux_x.v(n=n) - flux_x.ip(1, n=n)) + \
                        dtdy*(flux_y.v(n=n) - flux_y.jp(1, n=n))

            # gravitational source terms
            ymom.d[:,:] += 0.5*self.dt*old_dens.d[:,:]*grav
            ener.d[:,:] += 0.5*self.dt*old_ymom.d[:,:]*grav



        # increment the time
        self.cc_data.t += self.dt
        self.n += 1

        tm_evolve.end()
