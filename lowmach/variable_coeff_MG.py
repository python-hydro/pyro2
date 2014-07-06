import MG

class VarCoeffCCMG2d(MG.CellCenterMG2d):

    # this is a multigrid solver that supports variable coefficients
    #
    # we need to accept a coefficient array, coeffs, defined at each
    # level.  We can do this through an init_coeffs, and restrict it
    # down the MG grids once.
    #
    # we need a new compute_residual() and smooth() function, that
    # understands coeffs.
