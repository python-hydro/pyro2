import matplotlib

try:
    matplotlib.rcParams['mpl_toolkits.legacy_colorbar'] = False
except KeyError:
    pass
import matplotlib.pyplot as plt
import numpy as np

import pyro.compressible as compressible
import pyro.compressible.eos as eos
import pyro.util.plot_tools as plot_tools


class Simulation(compressible.Simulation):

    def initialize(self):
        """
        For the reacting compressible solver, our initialization of
        the data is the same as the compressible solver, but we
        supply additional variables.
        """
        super().initialize(extra_vars=["fuel", "ash"])

    def burn(self, dt):
        """ react fuel to ash """
        # compute T

        # compute energy generation rate

        # update energy due to reaction
        pass

    def diffuse(self, dt):
        """ diffuse for dt """

        # compute T

        # compute div kappa grad T

        # update energy due to diffusion
        pass

    def evolve(self):
        """
        Evolve the equations of compressible hydrodynamics through a
        timestep dt.
        """

        # we want to do Strang-splitting here
        self.burn(self.dt/2)

        self.diffuse(self.dt/2)

        if self.particles is not None:
            self.particles.update_particles(self.dt/2)

        # note: this will do the time increment and n increment
        super().evolve()

        if self.particles is not None:
            self.particles.update_particles(self.dt/2)

        self.diffuse(self.dt/2)

        self.burn(self.dt/2)

    def dovis(self):
        """
        Do runtime visualization.
        """

        plt.clf()

        plt.rc("font", size=10)

        # we do this even though ivars is in self, so this works when
        # we are plotting from a file
        ivars = compressible.Variables(self.cc_data)

        # access gamma from the cc_data object so we can use dovis
        # outside of a running simulation.
        gamma = self.cc_data.get_aux("gamma")

        q = compressible.cons_to_prim(self.cc_data.data, gamma, ivars, self.cc_data.grid)

        rho = q[:, :, ivars.irho]
        u = q[:, :, ivars.iu]
        v = q[:, :, ivars.iv]
        p = q[:, :, ivars.ip]
        e = eos.rhoe(gamma, p)/rho

        X = q[:, :, ivars.ix]

        magvel = np.sqrt(u**2 + v**2)

        myg = self.cc_data.grid

        fields = [rho, magvel, p, e, X]
        field_names = [r"$\rho$", r"U", "p", "e", r"$X_\mathrm{fuel}$"]

        f, axes, cbar_title = plot_tools.setup_axes(myg, len(fields))

        for n, ax in enumerate(axes):
            v = fields[n]

            img = ax.imshow(np.transpose(v.v()),
                            interpolation="nearest", origin="lower",
                            extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
                            cmap=self.cm)

            ax.set_xlabel("x")
            ax.set_ylabel("y")

            # needed for PDF rendering
            cb = axes.cbar_axes[n].colorbar(img)
            cb.formatter = matplotlib.ticker.FormatStrFormatter("")
            cb.solids.set_rasterized(True)
            cb.solids.set_edgecolor("face")

            if cbar_title:
                cb.ax.set_title(field_names[n])
            else:
                ax.set_title(field_names[n])

        plt.figtext(0.05, 0.0125, f"t = {self.cc_data.t:10.5g}")

        plt.pause(0.001)
        plt.draw()
