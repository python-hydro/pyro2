"""
This multigrid solver is build from multigrid/MG.py and implements
a variable coefficient solver for an equation of the form:

div { eta grad phi } = f

where eta is defined on the same grid as phi.

A cell-centered discretization is used throughout.
"""

import multigrid.MG as MG
import numpy
import sys

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
                 true_function=None):

       
        # initialize the MG object with the auxillary "coeffs" field
        MG.CellCenterMG2d.__init__(self, nx, ny, ng=1,
                                   xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                                   xl_BC_type=xl_BC_type, xr_BC_type=xr_BC_type,
                                   yl_BC_type=yl_BC_type, yr_BC_type=yr_BC_type,
                                   alpha=0.0, beta=0.0,
                                   nsmooth=nsmooth, nsmooth_bottom=nsmooth_bottom,
                                   verbose=verbose,
                                   aux_field="coeffs", aux_bc=coeffs_bc,
                                   true_function=true_function)


        # set the coefficients and restrict them down the hierarchy
        # we only need to do this once.
        c = self.grids[self.nlevels-1].get_var("coeffs")
        c[:,:] = coeffs.copy()

        self.grids[self.nlevels-1].fill_BC("coeffs")

        n = self.nlevels-2
        while n >= 0:

            f_patch = self.grids[n+1]
            c_patch = self.grids[n]

            coeffs_c = c_patch.get_var("coeffs")
            coeffs_c[:,:] = f_patch.restrict("coeffs")

            self.grids[n].fill_BC("coeffs")

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
        c = self.grids[level].get_var("coeffs")

        myg = self.grids[level].grid

        self.grids[level].fill_BC("v")


        eta_x = myg.scratch_array()
        eta_y = myg.scratch_array()

        # the eta's are defined on the interfaces, so 
        # eta_x[i,j] will be eta_{i-1/2,j} and 
        # eta_y[i,j] will be eta_{i,j-1/2}

        eta_x[myg.ilo:myg.ihi+2,myg.jlo:myg.jhi+1] = \
            0.5*(c[myg.ilo-1:myg.ihi+1,myg.jlo:myg.jhi+1] +
                 c[myg.ilo  :myg.ihi+2,myg.jlo:myg.jhi+1])

        eta_y[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+2] = \
            0.5*(c[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi+1] +
                 c[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+2])

        # print(c)
        # print(eta_x)
        # print(eta_y)
        # sys.exit()

        # eta_x[:,:] = 1.0
        # eta_y[:,:] = 1.0

        print( "min/max eta_x: ", numpy.min(eta_x), numpy.max(eta_x))
        print( "min/max eta_y: ", numpy.min(eta_y), numpy.max(eta_y))

        eta_x /= myg.dx**2
        eta_y /= myg.dy**2

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



    def _compute_residual(self, level):
        """ compute the residual and store it in the r variable"""

        v = self.grids[level].get_var("v")
        f = self.grids[level].get_var("f")
        r = self.grids[level].get_var("r")
        c = self.grids[level].get_var("coeffs")

        myg = self.grids[level].grid

        eta_x = myg.scratch_array()
        eta_y = myg.scratch_array()

        # the eta's are defined on the interfaces, so 
        # eta_x[i,j] will be eta_{i-1/2,j} and 
        # eta_y[i,j] will be eta_{i,j-1/2}

        eta_x[myg.ilo:myg.ihi+2,myg.jlo:myg.jhi+1] = \
            0.5*(c[myg.ilo-1:myg.ihi+1,myg.jlo:myg.jhi+1] +
                 c[myg.ilo  :myg.ihi+2,myg.jlo:myg.jhi+1])

        eta_y[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+2] = \
            0.5*(c[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi+1] +
                 c[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+2])

        eta_x /= myg.dx**2
        eta_y /= myg.dy**2

        # compute the residual 
        # r = f - L_eta phi
        L_eta_phi = ( 
            # x terms
            eta_x[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1]* \
            (v[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] - 
             v[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1]) - \
            #
            eta_x[myg.ilo  :myg.ihi+1,myg.jlo:myg.jhi+1]* \
            (v[myg.ilo  :myg.ihi+1,myg.jlo  :myg.jhi+1] -
             v[myg.ilo-1:myg.ihi  ,myg.jlo  :myg.jhi+1]) + \
            # y terms
            eta_y[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2]* \
            (v[myg.ilo:myg.ihi+1,myg.jlo+1:myg.jhi+2] -  # y-diff
             v[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1]) - \
            #
            eta_y[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1]* \
            (v[myg.ilo:myg.ihi+1,myg.jlo  :myg.jhi+1] -
             v[myg.ilo:myg.ihi+1,myg.jlo-1:myg.jhi  ]) )

        r[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
            f[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] - L_eta_phi



