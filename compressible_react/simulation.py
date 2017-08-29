from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid

import mesh.integration as integration
import compressible
import compressible.eos as eos

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


    def evolve(self):
        """
        Evolve the equations of compressible hydrodynamics through a
        timestep dt.
        """

        # we want to do Strang-splitting here
        self.burn(self.dt/2)

        # note: this will do the time increment and n increment
        super().evolve()

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

        rho = q[:,:,ivars.irho]
        u = q[:,:,ivars.iu]
        v = q[:,:,ivars.iv]
        p = q[:,:,ivars.ip]
        e = eos.rhoe(gamma, p)/rho
        X = q[:,:,ivars.ix]
        print("X: ", X.min(), X.max())

        magvel = np.sqrt(u**2 + v**2)

        myg = self.cc_data.grid

        # figure out the geometry
        L_x = self.cc_data.grid.xmax - self.cc_data.grid.xmin
        L_y = self.cc_data.grid.ymax - self.cc_data.grid.ymin

        f = plt.figure(1)

        cbar_title = False

        if L_x > 2*L_y:
            # we want 5 rows:
            axes = AxesGrid(f, 111,
                            nrows_ncols=(5, 1),
                            share_all=True,
                            cbar_mode="each",
                            cbar_location="top",
                            cbar_pad="10%",
                            cbar_size="25%",
                            axes_pad=(0.25, 0.65),
                            add_all=True, label_mode="L")
            cbar_title = True

        elif L_y > 2*L_x:
            # we want 5 columns:  rho  |U|  p  e
            axes = AxesGrid(f, 111,
                            nrows_ncols=(1, 5),
                            share_all=True,
                            cbar_mode="each",
                            cbar_location="right",
                            cbar_pad="10%",
                            cbar_size="25%",
                            axes_pad=(0.65, 0.25),
                            add_all=True, label_mode="L")

        else:
            # 3x2 grid of plots
            axes = AxesGrid(f, 111,
                            nrows_ncols=(3, 2),
                            share_all=True,
                            cbar_mode="each",
                            cbar_location="right",
                            cbar_pad="2%",
                            axes_pad=(0.65, 0.25),
                            add_all=True, label_mode="L")

        fields = [rho, magvel, p, e, X]
        field_names = [r"$\rho$", r"U", "p", "e", r"$X_\mathrm{fuel}$"]

        for n in range(4):
            ax = axes[n]

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
