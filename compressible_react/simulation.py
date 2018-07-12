from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import compressible
import compressible.eos as eos

import compressible_react.burning as burning

import util.plot_tools as plot_tools


class Simulation(compressible.Simulation):

    def initialize(self):
        """
        For the reacting compressible solver, our initialization of
        the data is the same as the compressible solver, but we
        supply additional variables.
        """
        super().initialize(extra_vars=["fuel", "ash"])

        myd = self.cc_data

    def burn(self, dt):
        """ react fuel to ash """

        e = self.cc_data.get_var("eint")
        ener = self.cc_data.get_var("eint")
        dens = self.cc_data.get_var("density")
        fuel = self.cc_data.get_var("fuel")
        ash = self.cc_data.get_var("ash")

        # compute T
        Cv = self.rp.get_param("eos.cv")
        temp = eos.temp(e, Cv)

        # compute energy generation rate
        omega_dot = burning.compute_energy_gen_rate(self.cc_data, temp)

        # update energy due to reaction
        ener[:,:] += dens * omega_dot * dt

        # update fuel and ash??

    def diffuse(self, dt):
        """ diffuse for dt """

        myg = self.cc_data.grid

        e = self.cc_data.get_var("eint")
        ener = self.cc_data.get_var("eint")

        # compute T
        Cv = self.rp.get_param("eos.cv")
        temp = eos.temp(e, Cv)

        # compute div kappa grad T
        k = self.rp.get_param("diffusion.k")

        div_kappa_grad_T = myg.scratch_array()
        div_kappa_grad_T.v()[:, :] = k * (
            (temp.ip(1) + temp.ip(-1) - 2.0*temp.v())/myg.dx**2 +
            (temp.jp(1) + temp.jp(-1) - 2.0*temp.v())/myg.dy**2)

        # update energy due to diffusion
        ener[:,:] += div_kappa_grad_T * dt

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
            cb.solids.set_rasterized(True)
            cb.solids.set_edgecolor("face")

            if cbar_title:
                cb.ax.set_title(field_names[n])
            else:
                ax.set_title(field_names[n])

        plt.figtext(0.05, 0.0125, "t = {:10.5g}".format(self.cc_data.t))

        plt.pause(0.001)
        plt.draw()
