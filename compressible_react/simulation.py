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


    def evolve(self):
        """
        Evolve the equations of compressible hydrodynamics through a
        timestep dt.
        """

        # we want to do Strang-splitting here

        # call the burner

        # call the super class evolve to advance hydro

        # call the burner


        # increment the time
        myd.t += self.dt
        self.n += 1

        tm_evolve.end()
