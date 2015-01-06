"""
This multigrid solver is build from multigrid/MG.py and implements
a variable coefficient solver for an equation of the form:

div { eta grad phi } = f

where eta is defined on the same grid as phi.

A cell-centered discretization is used throughout.
"""

from __future__ import print_function

import multigrid.MG as MG
import numpy
import pylab

numpy.set_printoptions(precision=3, linewidth=128)


class _EdgeCoeffs:
    """
    a simple container class to hold the edge-centered coefficients
    """
    def __init__(self, g, eta, empty=False):

        self.grid = g

        if not empty:
            eta_x = g.scratch_array()
            eta_y = g.scratch_array()

            # the eta's are defined on the interfaces, so 
            # eta_x[i,j] will be eta_{i-1/2,j} and 
            # eta_y[i,j] will be eta_{i,j-1/2}

            eta_x[g.ilo:g.ihi+2,g.jlo:g.jhi+1] = \
                0.5*(eta[g.ilo-1:g.ihi+1,g.jlo:g.jhi+1] +
                     eta[g.ilo  :g.ihi+2,g.jlo:g.jhi+1])

            eta_y[g.ilo:g.ihi+1,g.jlo:g.jhi+2] = \
                0.5*(eta[g.ilo:g.ihi+1,g.jlo-1:g.jhi+1] +
                     eta[g.ilo:g.ihi+1,g.jlo  :g.jhi+2])

            eta_x /= g.dx**2
            eta_y /= g.dy**2

            self.x = eta_x
            self.y = eta_y


    def restrict(self):
        """
        restrict the edge values to a coarser grid.  Return a new
        _EdgeCoeffs object
        """

        cg = self.grid.coarse_like(2)

        c_edge_coeffs = _EdgeCoeffs(cg, None, empty=True)

        c_eta_x = cg.scratch_array()
        c_eta_y = cg.scratch_array()

        fg = self.grid
        
        c_eta_x[cg.ilo:cg.ihi+2,cg.jlo:cg.jhi+1] = \
            0.5*(self.x[fg.ilo:fg.ihi+2:2,fg.jlo  :fg.jhi+1:2] +
                 self.x[fg.ilo:fg.ihi+2:2,fg.jlo+1:fg.jhi+1:2]) 
        
        # redo the normalization
        c_edge_coeffs.x = c_eta_x*fg.dx**2/cg.dx**2

        c_eta_y[cg.ilo:cg.ihi+1,cg.jlo:cg.jhi+2] = \
            0.5*(self.y[fg.ilo  :fg.ihi+1:2,fg.jlo:fg.jhi+2:2] +
                 self.y[fg.ilo+1:fg.ihi+1:2,fg.jlo:fg.jhi+2:2]) 
        
        c_edge_coeffs.y = c_eta_y*fg.dy**2/cg.dy**2
        
        return c_edge_coeffs

        
class VarCoeffCCMG2d(MG.CellCenterMG2d):
    """
    this is a multigrid solver that supports variable coefficients
    
    we need to accept a coefficient array, coeffs, defined at each
    level.  We can do this at the fine level and restrict it
    down the MG grids once.
    
    we need a new compute_residual() and smooth() function, that
    understands coeffs.
    """
   
    def __init__(self, nx, ny, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
                 xl_BC_type="dirichlet", xr_BC_type="dirichlet",
                 yl_BC_type="dirichlet", yr_BC_type="dirichlet",
                 nsmooth=10, nsmooth_bottom=50,
                 verbose=0, 
                 coeffs=None, coeffs_bc=None,
                 true_function=None, vis=0, vis_title=""):

        # we'll keep a list of the coefficients averaged to the interfaces
        # on each level
        self.edge_coeffs = []
       
        # initialize the MG object with the auxillary "coeffs" field
        MG.CellCenterMG2d.__init__(self, nx, ny, ng=1,
                                   xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                                   xl_BC_type=xl_BC_type, xr_BC_type=xr_BC_type,
                                   yl_BC_type=yl_BC_type, yr_BC_type=yr_BC_type,
                                   alpha=0.0, beta=0.0,
                                   nsmooth=nsmooth, nsmooth_bottom=nsmooth_bottom,
                                   verbose=verbose,
                                   aux_field="coeffs", aux_bc=coeffs_bc,
                                   true_function=true_function, vis=vis, 
                                   vis_title=vis_title)


        # set the coefficients and restrict them down the hierarchy
        # we only need to do this once.
        c = self.grids[self.nlevels-1].get_var("coeffs")
        c[:,:] = coeffs.copy()
        
        self.grids[self.nlevels-1].fill_BC("coeffs")

        # put the coefficients on edges
        self.edge_coeffs.insert(0, _EdgeCoeffs(self.grids[self.nlevels-1].grid, c))

        n = self.nlevels-2
        while n >= 0:

            # create the edge coefficients on level n by restricting from the
            # finer grid
            f_patch = self.grids[n+1]
            c_patch = self.grids[n]

            coeffs_c = c_patch.get_var("coeffs")
            coeffs_c[:,:] = f_patch.restrict("coeffs")

            self.grids[n].fill_BC("coeffs")

            # put the coefficients on edges
            self.edge_coeffs.insert(0, self.edge_coeffs[0].restrict())  #_EdgeCoeffs(self.grids[n].grid, coeffs_c))

            # if we are periodic, then we should force the edge coefficents
            # to be periodic
            # if self.grids[n].BCs["coeffs"].xlb == "periodic":
            #     self.edge_coeffs[0].x[self.grids[n].grid.ihi+1,:] = \
            #         self.edge_coeffs[0].x[self.grids[n].grid.ilo,:]

            # if self.grids[n].BCs["coeffs"].ylb == "periodic":
            #     self.edge_coeffs[0].y[:,self.grids[n].grid.jhi+1] = \
            #         self.edge_coeffs[0].y[:,self.grids[n].grid.jlo]

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

        self.grids[level].fill_BC("v")

        eta_x = self.edge_coeffs[level].x
        eta_y = self.edge_coeffs[level].y

        # print( "min/max c: {}, {}".format(numpy.min(c), numpy.max(c)))
        # print( "min/max eta_x: {}, {}".format(numpy.min(eta_x), numpy.max(eta_x)))
        # print( "min/max eta_y: {}, {}".format(numpy.min(eta_y), numpy.max(eta_y)))


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

            for n, (ix, iy) in enumerate([(0,0), (1,1), (1,0), (0,1)]):

                denom = (
                    eta_x[myg.ilo+1+ix:myg.ihi+2:2,
                          myg.jlo+iy  :myg.jhi+1:2] +
                    #
                    eta_x[myg.ilo+ix  :myg.ihi+1:2,
                          myg.jlo+iy  :myg.jhi+1:2] +
                    #
                    eta_y[myg.ilo+ix  :myg.ihi+1:2,
                          myg.jlo+1+iy:myg.jhi+2:2] +
                    #
                    eta_y[myg.ilo+ix  :myg.ihi+1:2,
                          myg.jlo+iy  :myg.jhi+1:2])

                v[myg.ilo+ix:myg.ihi+1:2,myg.jlo+iy:myg.jhi+1:2] = (
                    -f[myg.ilo+ix:myg.ihi+1:2,myg.jlo+iy:myg.jhi+1:2] +
                    # eta_{i+1/2,j} phi_{i+1,j}
                    eta_x[myg.ilo+1+ix:myg.ihi+2:2,
                          myg.jlo+iy  :myg.jhi+1:2] *
                    v[myg.ilo+1+ix:myg.ihi+2:2,
                      myg.jlo+iy  :myg.jhi+1:2] +
                    # eta_{i-1/2,j} phi_{i-1,j}                 
                    eta_x[myg.ilo+ix:myg.ihi+1:2,
                          myg.jlo+iy:myg.jhi+1:2]*
                    v[myg.ilo-1+ix:myg.ihi  :2,
                      myg.jlo+iy  :myg.jhi+1:2] +
                    # eta_{i,j+1/2} phi_{i,j+1}
                    eta_y[myg.ilo+ix:myg.ihi+1:2,
                          myg.jlo+1+iy:myg.jhi+2:2]*
                    v[myg.ilo+ix  :myg.ihi+1:2,
                      myg.jlo+1+iy:myg.jhi+2:2] +
                    # eta_{i,j-1/2} phi_{i,j-1}
                    eta_y[myg.ilo+ix:myg.ihi+1:2,
                          myg.jlo+iy:myg.jhi+1:2]*
                    v[myg.ilo+ix  :myg.ihi+1:2,
                      myg.jlo-1+iy:myg.jhi  :2]) / denom

            
                if n == 1 or n == 3:
                    self.grids[level].fill_BC("v")


            if self.vis == 1:
                pylab.clf()

                pylab.subplot(221)
                self._draw_solution()

                pylab.subplot(222)        
                self._draw_V()

                pylab.subplot(223)        
                self._draw_main_solution()

                pylab.subplot(224)        
                self._draw_main_error()


                pylab.suptitle(self.vis_title, fontsize=18)

                pylab.draw()
                pylab.savefig("mg_%4.4d.png" % (self.frame))
                self.frame += 1


    def _compute_residual(self, level):
        """ compute the residual and store it in the r variable"""


        v = self.grids[level].get_var("v")
        f = self.grids[level].get_var("f")
        r = self.grids[level].get_var("r")

        myg = self.grids[level].grid

        eta_x = self.edge_coeffs[level].x
        eta_y = self.edge_coeffs[level].y


        # compute the residual 
        # r = f - L_eta phi
        L_eta_phi = ( 
            # eta_{i+1/2,j} (phi_{i+1,j} - phi_{i,j})
            eta_x[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1]* \
            (v[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] - 
             v[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1]) - \
            # eta_{i-1/2,j} (phi_{i,j} - phi_{i-1,j})
            eta_x[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1]* \
            (v[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1] -
             v[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1]) + \
            # eta_{i,j+1/2} (phi_{i,j+1} - phi_{i,j})
            eta_y[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2]* \
            (v[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -  # y-diff
             v[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1]) - \
            # eta_{i,j-1/2} (phi_{i,j} - phi_{i,j-1})
            eta_y[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1]* \
            (v[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1] -
             v[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ]) )

        r[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
            f[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] - L_eta_phi



