r"""
This multigrid solver is build from multigrid/MG.py
and implements a more general solver for an equation of the form

.. math::

   \alpha \phi + \nabla\cdot { \beta \nabla \phi } + \gamma \cdot \nabla \phi = f

where alpha, beta, and gamma are defined on the same grid as phi.
These should all come in as cell-centered quantities.  The solver
will put beta on edges.  Note that gamma is a vector here, with
x- and y-components.

A cell-centered discretization for phi is used throughout.
"""


import matplotlib.pyplot as plt
import numpy as np

import pyro.multigrid.edge_coeffs as ec
import pyro.multigrid.MG as MG

np.set_printoptions(precision=3, linewidth=128)


class GeneralMG2d(MG.CellCenterMG2d):
    r"""
    this is a multigrid solver that supports our general elliptic
    equation.

    we need to accept a coefficient ``CellCenterData2d`` object with
    fields defined for ``alpha``, ``beta``, ``gamma_x``, and ``gamma_y`` on the
    fine level.

    We then restrict this data through the MG hierarchy (and
    average beta to the edges).

    we need a ``new compute_residual()`` and ``smooth()`` function, that
    understands these coeffs.
    """

    def __init__(self, nx, ny, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
                 xl_BC_type="dirichlet", xr_BC_type="dirichlet",
                 yl_BC_type="dirichlet", yr_BC_type="dirichlet",
                 xl_BC=None, xr_BC=None,
                 yl_BC=None, yr_BC=None,
                 nsmooth=10, nsmooth_bottom=50,
                 verbose=0,
                 coeffs=None,
                 true_function=None, vis=0, vis_title=""):
        """
        here, coeffs is a CCData2d object
        """

        # we'll keep a list of the beta coefficients averaged to the
        # interfaces on each level -- note: these will already be
        # scaled by 1/dx**2
        self.beta_edge = []

        # initialize the MG object with the auxillary fields
        MG.CellCenterMG2d.__init__(self, nx, ny, ng=1,
                                   xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                                   xl_BC_type=xl_BC_type, xr_BC_type=xr_BC_type,
                                   yl_BC_type=yl_BC_type, yr_BC_type=yr_BC_type,
                                   xl_BC=xl_BC, xr_BC=xr_BC,
                                   yl_BC=yl_BC, yr_BC=yr_BC,
                                   alpha=0.0, beta=0.0,
                                   nsmooth=nsmooth, nsmooth_bottom=nsmooth_bottom,
                                   verbose=verbose,
                                   aux_field=["alpha", "beta", "gamma_x", "gamma_y"],
                                   aux_bc=[coeffs.BCs["alpha"], coeffs.BCs["beta"],
                                           coeffs.BCs["gamma_x"], coeffs.BCs["gamma_y"]],
                                   true_function=true_function, vis=vis,
                                   vis_title=vis_title)

        # the coefficents come in a dictionary.  Set the coefficients
        # and restrict them down the hierarchy we only need to do this
        # once.  We need to hold the original coeffs in our grid so we
        # can do a ghost cell fill.
        for c in ["alpha", "beta", "gamma_x", "gamma_y"]:
            v = self.grids[self.nlevels-1].get_var(c)
            v.v()[:, :] = coeffs.get_var(c).v()

            self.grids[self.nlevels-1].fill_BC(c)

            n = self.nlevels-2
            while n >= 0:
                f_patch = self.grids[n+1]
                c_patch = self.grids[n]

                coeffs_c = c_patch.get_var(c)
                coeffs_c.v()[:, :] = f_patch.restrict(c).v()

                self.grids[n].fill_BC(c)
                n -= 1

        # put the beta coefficients on edges
        beta = self.grids[self.nlevels-1].get_var("beta")
        self.beta_edge.insert(0, ec.EdgeCoeffs(self.grids[self.nlevels-1].grid, beta))

        n = self.nlevels-2
        while n >= 0:
            self.beta_edge.insert(0, self.beta_edge[0].restrict())
            n -= 1

    def smooth(self, level, nsmooth):
        """
        Use red-black Gauss-Seidel iterations to smooth the solution
        at a given level.  This is used at each stage of the V-cycle
        (up and down) in the MG solution, but it can also be called
        directly to solve the elliptic problem (although it will take
        many more iterations).

        Parameters
        ----------
        level : int
            The level in the MG hierarchy to smooth the solution
        nsmooth : int
            The number of r-b Gauss-Seidel smoothing iterations to perform

        """
        v = self.grids[level].get_var("v")
        f = self.grids[level].get_var("f")

        myg = self.grids[level].grid

        dx = myg.dx
        dy = myg.dy

        self.grids[level].fill_BC("v")

        alpha = self.grids[level].get_var("alpha")
        gamma_x = 0.5*self.grids[level].get_var("gamma_x")/dx
        gamma_y = 0.5*self.grids[level].get_var("gamma_y")/dy

        # these are already scaled by 1/dx**2 in the EdgeCoeffs
        # construction
        beta_x = self.beta_edge[level].x
        beta_y = self.beta_edge[level].y

        # do red-black G-S
        for i in range(nsmooth):

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

                denom = (
                    alpha.ip_jp(ix, iy, s=2) -
                    beta_x.ip_jp(1+ix, iy, s=2) - beta_x.ip_jp(ix, iy, s=2) -
                    beta_y.ip_jp(ix, 1+iy, s=2) - beta_y.ip_jp(ix, iy, s=2))

                v.ip_jp(ix, iy, s=2)[:, :] = (f.ip_jp(ix, iy, s=2) -
                    # (beta_{i+1/2,j} + gamma^x_{i,j}) phi_{i+1,j}
                    (beta_x.ip_jp(1+ix, iy, s=2) + gamma_x.ip_jp(ix, iy, s=2)) * v.ip_jp(1+ix, iy, s=2) -
                    # (beta_{i-1/2,j} - gamma^x_{i,j}) phi_{i-1,j}
                    (beta_x.ip_jp(ix, iy, s=2) - gamma_x.ip_jp(ix, iy, s=2)) * v.ip_jp(-1+ix, iy, s=2) -
                    # (beta_{i,j+1/2} + gamma^y_{i,j}) phi_{i,j+1}
                    (beta_y.ip_jp(ix, 1+iy, s=2) + gamma_y.ip_jp(ix, iy, s=2)) * v.ip_jp(ix, 1+iy, s=2) -
                    # (beta_{i,j-1/2} - gamma^y_{i,j}) phi_{i,j-1}
                    (beta_y.ip_jp(ix, iy, s=2) - gamma_y.ip_jp(ix, iy, s=2)) * v.ip_jp(ix, -1+iy, s=2)) / denom

                if n == 1 or n == 3:
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

        dx = myg.dx
        dy = myg.dy

        alpha = self.grids[level].get_var("alpha")
        gamma_x = 0.5*self.grids[level].get_var("gamma_x")/dx
        gamma_y = 0.5*self.grids[level].get_var("gamma_y")/dy

        # these already have a 1/dx**2 scaling in them
        beta_x = self.beta_edge[level].x
        beta_y = self.beta_edge[level].y

        # compute the residual
        # r = f - L_eta phi
        L_eta_phi = (
            # alpha_{i,j} phi_{i,j}
            alpha.v()*v.v() +
            # beta_{i+1/2,j} (phi_{i+1,j} - phi_{i,j})
            beta_x.ip(1)*(v.ip(1) - v.v()) - \
            # beta_{i-1/2,j} (phi_{i,j} - phi_{i-1,j})
            beta_x.v()*(v.v() - v.ip(-1)) + \
            # beta_{i,j+1/2} (phi_{i,j+1} - phi_{i,j})
            beta_y.jp(1)*(v.jp(1) - v.v()) - \
            # beta_{i,j-1/2} (phi_{i,j} - phi_{i,j-1})
            beta_y.v()*(v.v() - v.jp(-1)) + \
            # gamma^x_{i,j} (phi_{i+1,j} - phi_{i-1,j})
            gamma_x.v()*(v.ip(1) - v.ip(-1)) + \
            # gamma^y_{i,j} (phi_{i,j+1} - phi_{i,j-1})
            gamma_y.v()*(v.jp(1) - v.jp(-1)))

        r.v()[:, :] = f.v() - L_eta_phi
