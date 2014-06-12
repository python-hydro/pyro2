"""
This is a gamma-law equation of state.  
"""

def pres(gamma, dens, eint):
    """ given the density and the specific internal energy, return the
        pressure """
    p = dens*eint*(gamma - 1.0)
    return p


def dens(gamma, pres, eint):
    """ given the pressure and the specific internal energy, return
        the density """
    dens = pres/(eint*(gamma - 1.0))
    return dens


def rhoe(gamma, pres):
    """ given the pressure, return (rho * e) """
    rhoe = pres/(gamma - 1.0)
    return rhoe
