from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import compressible
import compressible.eos as eos

import compressible_react.burning as burning

import util.plot_tools as plot_tools
from util import msg


class Simulation(compressible.Simulation):

    def initialize(self):
        """
        For the reacting compressible solver, our initialization of
        the data is the same as the compressible solver, but we
        supply additional variables.
        """

        network = self.rp.get_param("network.network_type")

        if network == "powerlaw":
            self.network = burning.PowerLaw()
        elif network == "null":
            self.network = burning.Network(nspec_evolve=2)
        else:
            msg.fail("ERROR: Do not recognise network %s" % network)

        self.network.initialize(self.rp)

        super().initialize(extra_vars=self.network.spec_names)

    def burn(self, dt):
        """ react fuel to ash """

        ener = self.cc_data.get_var("energy")
        dens = self.cc_data.get_var("density")
        fuel = self.cc_data.get_var("fuel")
        ash = self.cc_data.get_var("ash")

        # compute energy generation rate
        H, omega_dot = self.network.energy_and_species_creation(self.cc_data)

        # update energy due to reaction
        ener[:, :] += dens * H * dt

        fuel[:, :] += omega_dot[:, :, 0] * dt * dens
        ash[:, :] += omega_dot[:, :, 1] * dt * dens

        # enforce 0 <= X_k <= 1
        fuel[fuel < 0] = 0
        ash[ash < 0] = 0

        fuel[fuel > dens] = dens[fuel > dens]
        ash[ash > dens] = dens[ash > dens]

        # enforce sum_k X_k = 1 by renormalizing
        sum = (fuel + ash) / dens
        fuel[:, :] /= sum
        ash[:, :] /= sum

    def diffuse(self, dt):
        """ diffuse for dt """

        myg = self.cc_data.grid

        e = self.cc_data.get_var("eint")
        ener = self.cc_data.get_var("energy")

        # compute T
        Cv = self.rp.get_param("eos.cv")
        temp = eos.temp(e, Cv)

        # compute div kappa grad T
        k_const = self.rp.get_param("diffusion.k")
        const_opacity = self.rp.get_param("diffusion.constant_kappa")
        k = burning.k_th(self.cc_data, temp, k_const, const_opacity)

        div_kappa_grad_T = myg.scratch_array()

        div_kappa_grad_T.v()[:, :] = 0.25 * (
            (k.ip(1) * (temp.ip(2) - temp.v()) -
             k.ip(-1) * (temp.v() - temp.ip(-2))) / myg.dx**2 +
            (k.jp(1) * (temp.jp(2) - temp.v()) -
             k.jp(-1) * (temp.v() - temp.jp(-2))) / myg.dy**2)

        # update energy due to diffusion
        ener[:, :] += div_kappa_grad_T * dt

    def evolve(self):
        """
        Evolve the equations of compressible hydrodynamics through a
        timestep dt.
        """

        # we want to do Strang-splitting here
        self.burn(self.dt / 2)

        self.diffuse(self.dt / 2)

        if self.particles is not None:
            self.particles.update_particles(self.dt / 2)

        # note: this will do the time increment and n increment
        super().evolve()

        if self.particles is not None:
            self.particles.update_particles(self.dt / 2)

        self.diffuse(self.dt / 2)

        self.burn(self.dt / 2)

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

        q = compressible.cons_to_prim(
            self.cc_data.data, gamma, ivars, self.cc_data.grid)

        rho = q[:, :, ivars.irho]
        u = q[:, :, ivars.iu]
        v = q[:, :, ivars.iv]
        p = q[:, :, ivars.ip]
        e = eos.rhoe(gamma, p) / rho

        X = q[:, :, ivars.ix]

        magvel = np.sqrt(u**2 + v**2)

        myg = self.cc_data.grid

        fields = [rho, magvel, u, p, e, X]
        field_names = [r"$\rho$", r"U", "u", "p", "e", r"$X_\mathrm{fuel}$"]

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
            cb.solids.set_rasterized(True)
            cb.solids.set_edgecolor("face")

            if cbar_title:
                cb.ax.set_title(field_names[n])
            else:
                ax.set_title(field_names[n])

        plt.figtext(0.05, 0.0125, "t = {:10.5g}".format(self.cc_data.t))

        plt.pause(0.001)
        plt.draw()
