r"""
A multigrid solver for axisymmetric coordinates (r-z) to solve
an elliptic equation of the form:

L \phi = f

where L is the Laplacian operator in cylindrical coords.
"""

import matplotlib.pyplot as plt
import numpy as np
import pyro.mesh.array_indexer as ai
from pyro.multigrid import MG

np.set_printoptions(precision=3, linewidth=128)


class AxisymmetricMG2d(MG.CellCenterMG2d):

    def __init__(self, nx, ny, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
                 xl_BC_type="reflect", xr_BC_type="dirichlet",
                 yl_BC_type="dirichlet", yr_BC_type="dirichlet",
                 xl_BC=None, xr_BC=None,
                 yl_BC=None, yr_BC=None,
                 nsmooth=10, nsmooth_bottom=50,
                 verbose=0,
                 true_function=None, vis=0, vis_title=""):

        # initialize the MG object with the auxillary "coeffs" field
        MG.CellCenterMG2d.__init__(self, nx, ny, ng=1,
                                   xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                                   xl_BC_type=xl_BC_type, xr_BC_type=xr_BC_type,
                                   yl_BC_type=yl_BC_type, yr_BC_type=yr_BC_type,
                                   xl_BC=xl_BC, xr_BC=xr_BC,
                                   yl_BC=yl_BC, yr_BC=yr_BC,
                                   nsmooth=nsmooth, nsmooth_bottom=nsmooth_bottom,
                                   verbose=verbose,
                                   true_function=true_function, vis=vis,
                                   vis_title=vis_title)

    def smooth(self, level, nsmooth):
        """
        Use red-black Gauss-Seidel iterations
        """

        v = self.grids[level].get_var("v")
        f = self.grids[level].get_var("f")

        self.grids[level].fill_BC("v")

        myg = self.grids[level].grid

        dr = myg.dx
        dz = myg.dy

        rc = ai.ArrayIndexer(myg.x2d, grid=myg)
        rl = rc - 0.5 * dr
        rr = rc + 0.5 * dr

        denom = 2 * (1.0 / dr**2 + 1.0 / dz**2)

        # do red-black G-S
        for _ in range(nsmooth):

            # do the red black updating in four decoupled groups
            #
            #
            #        |       |       |
            #      --+-------+-------+--
            #        |       |       |
            #        |   4   |   3   |
            #        |       |       |
            #      --+-------+-------+--
            #        |       |       |
            #   jlo  |   1   |   2   |
            #        |       |       |
            #      --+-------+-------+--
            #        |  ilo  |       |
            #
            # groups 1 and 3 are done together, then we need to
            # fill ghost cells, and then groups 2 and 4

            for n, (ix, iy) in enumerate([(0, 0), (1, 1), (1, 0), (0, 1)]):

                v.ip_jp(ix, iy, s=2)[:, :] = (-f.ip_jp(ix, iy, s=2) +
                    1.0 / (rc.ip(ix, s=2) * dr**2) * (rr.ip(ix, s=2) * v.ip_jp(1+ix, iy, s=2) +
                                                      rl.ip(ix, s=2) * v.ip_jp(-1+ix, iy, s=2)) +
                    1.0 / dz**2 * (v.ip_jp(ix, 1+iy, s=2) + v.ip_jp(ix, -1+iy, s=2))) / denom

                if n in (1, 3):
                    self.grids[level].fill_BC("v")

            if self.vis == 1:
                plt.clf()

                plt.subplot(221)
                self._draw_solution()

                plt.subplot(222)
                self._draw_V()

                plt.subplot(223)
                self._draw_main_solution()

                plt.subplot(224)
                self._draw_main_error()

                plt.suptitle(self.vis_title, fontsize=18)

                plt.draw()
                plt.savefig("mg_%4.4d.png" % (self.frame))
                self.frame += 1

    def _compute_residual(self, level):
        """ compute the residual and store it in the r variable"""

        v = self.grids[level].get_var("v")
        f = self.grids[level].get_var("f")
        r = self.grids[level].get_var("r")

        myg = self.grids[level].grid

        dr = myg.dx
        dz = myg.dy

        rc = ai.ArrayIndexer(myg.x2d, grid=myg)
        rl = rc - 0.5 * dr
        rr = rc + 0.5 * dr

        # compute the residual
        # r = f - L_eta phi
        L_phi = (1.0 / (rc.v() * dr**2) * (rr.v() * v.ip(1) - 2.0 * rc.v() * v.v() + rl.v() * v.ip(-1)) +
                 1.0 / (dz**2) * (v.jp(1) - 2.0 * v.v() + v.jp(-1)))

        r.v()[:, :] = f.v() - L_phi
