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

        dens = self.cc_data.get_var("density")
        ymom = self.cc_data.get_var("y-momentum")
        ener = self.cc_data.get_var("energy")

        grav = self.rp.get_param("compressible.grav")

        myg = self.cc_data.grid

        Flux_x, Flux_y = fluxes(self.cc_data, self.aux_data, self.rp, 
                                self.vars, self.solid, self.tc, self.dt)

        old_dens = dens.copy()
        old_ymom = ymom.copy()

        # conservative update
        dtdx = self.dt/myg.dx
        dtdy = self.dt/myg.dy

        for n in range(self.vars.nvar):
            var = self.cc_data.get_var_by_index(n)

            var.v()[:,:] += \
                dtdx*(Flux_x.v(n=n) - Flux_x.ip(1, n=n)) + \
                dtdy*(Flux_y.v(n=n) - Flux_y.jp(1, n=n))

        # gravitational source terms
        ymom.d[:,:] += 0.5*self.dt*(dens.d[:,:] + old_dens.d[:,:])*grav
        ener.d[:,:] += 0.5*self.dt*(ymom.d[:,:] + old_ymom.d[:,:])*grav

        # increment the time
        self.cc_data.t += self.dt
        self.n += 1

        tm_evolve.end()
