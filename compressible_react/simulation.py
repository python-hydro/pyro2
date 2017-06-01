from __future__ import print_function

import numpy as np

import mesh.integration as integration
import compressible

class Simulation(compressible.Simulation):

    def initialize(self):
        """
        For the reacting compressible solver, our initialization of
        the data is the same as the compressible solver, but we 
        supply additional variables.
        """
        super().initialize(extra_vars=["fuel", "ash"])


    def burn(self, dt):
        pass


    def evolve(self):
        """
        Evolve the equations of compressible hydrodynamics through a
        timestep dt.
        """

        # we want to do Strang-splitting here
        self.burn(self.dt/2)

        super().evolve()

        self.burn(self.dt/2)


        # increment the time
        self.cc_data.t += self.dt
        self.n += 1

