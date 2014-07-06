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
                 alpha=0.0, beta=-1.0,
                 nsmooth=10, nsmooth_bottom=50,
                 verbose=0, 
                 coeffs=None, coeffs_bc=None,
                 true_function=None):

       
       # initialize the MG object with the auxillary "coeffs" field
       MG.__init__(self, nx, ny, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                   xl_BC_type=xl_BC_type, xr_BC_type=xr_BC_type,
                   yl_BC_type=yl_BC_type, yr_BC_type=yr_BC_type,
                   alpha=alpha, beta=beta,
                   nsmooth=nsmooth, nsmooth_bottom=nsmooth_bottom,
                   verbose=verbose,
                   aux_field="coeffs", aux_bc=coeffs_bc,
                   true_function=true_function)


       # set the coefficients and restrict them down the hierarchy
       # we only need to do this once.
       c = self.grids[self.nlevels].get_var("coeffs")
       c[:,:] = coeffs.copy()

       n = self.nlevels-1
       while n >= 1:

          f_patch = self.grids[n+1]
          c_patch = self.grids[n]

          coeffs_c = c_patch.get_var("coeffs")
          coeffs_c[:,:] = f_patch.restrict("coeffs")

          n -= 1
