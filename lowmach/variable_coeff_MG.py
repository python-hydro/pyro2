"""
This multigrid solver is build from multigrid/MG.py and implements
a variable coefficient solver for an equation of the form:

div { eta grad phi } = f

where eta is defined on the same grid as phi.

A cell-centered discretization is used throughout.
"""

import multigrid.MG as MG


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
                 alpha=0.0, 
                 nsmooth=10, nsmooth_bottom=50,
                 verbose=0, 
                 coeffs=None, coeffs_bc=None,
                 true_function=None):

       
        # initialize the MG object with the auxillary "coeffs" field
        MG.__init__(self, nx, ny, ng=1,
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
        c = self.grids[self.nlevels].get_var("coeffs")
        c[:,:] = coeffs.copy()

        self.grids[nlevels].fill_BC("coeffs")

        n = self.nlevels-1
        while n >= 1:

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

        eta_x[myg.ilo:myg.ihi+2,myg.jlo:myg.jhi+1] = \
            0.5*(c[myg:ilo-1:myg.ihi+1,myg.jlo:myg.jhi+1] +
                 c[myg:ilo  :myg.ihi+2,myg.jlo:myg.jhi+1])

        eta_y[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+2] = \
            0.5*(c[myg:ilo:myg.ihi+1,myg.jlo-1:myg.jhi+1] +
                 c[myg:ilo:myg.ihi+1,myg.jlo  :myg.jhi+2])

        eta_x /= myg.dx**2
        eta_y /= myg.dy**2

        # do red-black G-S
        i = 0
        while i < nsmooth:


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

            v[myg.ilo:myg.ihi+1:2,myg.jlo:myg.jhi+1:2] = \
                (f[myg.ilo:myg.ihi+1:2,myg.jlo:myg.jhi+1:2] +
                 xcoeff*(v[myg.ilo+1:myg.ihi+2:2,myg.jlo  :myg.jhi+1:2] +
                         v[myg.ilo-1:myg.ihi  :2,myg.jlo  :myg.jhi+1:2]) +
                 ycoeff*(v[myg.ilo  :myg.ihi+1:2,myg.jlo+1:myg.jhi+2:2] +
                         v[myg.ilo  :myg.ihi+1:2,myg.jlo-1:myg.jhi  :2])) / \
                (self.alpha + 2.0*xcoeff + 2.0*ycoeff)

            v[myg.ilo+1:myg.ihi+1:2,myg.jlo+1:myg.jhi+1:2] = \
                (f[myg.ilo+1:myg.ihi+1:2,myg.jlo+1:myg.jhi+1:2] +
                 xcoeff*(v[myg.ilo+2:myg.ihi+2:2,myg.jlo+1:myg.jhi+1:2] +
                         v[myg.ilo  :myg.ihi  :2,myg.jlo+1:myg.jhi+1:2]) +
                 ycoeff*(v[myg.ilo+1:myg.ihi+1:2,myg.jlo+2:myg.jhi+2:2] +
                         v[myg.ilo+1:myg.ihi+1:2,myg.jlo  :myg.jhi  :2])) / \
                (self.alpha + 2.0*xcoeff + 2.0*ycoeff)
            
            self.grids[level].fill_BC("v")
                                                     
            v[myg.ilo+1:myg.ihi+1:2,myg.jlo:myg.jhi+1:2] = \
                (f[myg.ilo+1:myg.ihi+1:2,myg.jlo:myg.jhi+1:2] +
                 xcoeff*(v[myg.ilo+2:myg.ihi+2:2,myg.jlo  :myg.jhi+1:2] +
                         v[myg.ilo  :myg.ihi  :2,myg.jlo  :myg.jhi+1:2]) +
                 ycoeff*(v[myg.ilo+1:myg.ihi+1:2,myg.jlo+1:myg.jhi+2:2] +
                         v[myg.ilo+1:myg.ihi+1:2,myg.jlo-1:myg.jhi  :2])) / \
                (self.alpha + 2.0*xcoeff + 2.0*ycoeff)

            v[myg.ilo:myg.ihi+1:2,myg.jlo+1:myg.jhi+1:2] = \
                (f[myg.ilo:myg.ihi+1:2,myg.jlo+1:myg.jhi+1:2] +
                 xcoeff*(v[myg.ilo+1:myg.ihi+2:2,myg.jlo+1:myg.jhi+1:2] +
                         v[myg.ilo-1:myg.ihi  :2,myg.jlo+1:myg.jhi+1:2]) +
                 ycoeff*(v[myg.ilo  :myg.ihi+1:2,myg.jlo+2:myg.jhi+2:2] +
                         v[myg.ilo  :myg.ihi+1:2,myg.jlo  :myg.jhi  :2])) / \
                (self.alpha + 2.0*xcoeff + 2.0*ycoeff)


            self.grids[level].fill_BC("v")

            i += 1
