"""
This is a gamma-law equation of state: p = rho e (gamma - 1), where
gamma is the constant ratio of specific heats.
"""


def pres(gamma, dens, eint):
    """
    Given the density and the specific internal energy, return the
    pressure

    Parameters
    ----------
    gamma : float
        The ratio of specific heats
    dens : float
        The density
    eint : float
        The specific internal energy

    Returns
    -------
    out : float
       The pressure

     """
    p = dens*eint*(gamma - 1.0)
    return p


def dens(gamma, pres, eint):
    """
    Given the pressure and the specific internal energy, return
    the density

    Parameters
    ----------
    gamma : float
        The ratio of specific heats
    pres : float
        The pressure
    eint : float
        The specific internal energy

    Returns
    -------
    out : float
       The density

    """
    dens = pres/(eint*(gamma - 1.0))
    return dens


def rhoe(gamma, pres):
    """
    Given the pressure, return (rho * e)

    Parameters
    ----------
    gamma : float
        The ratio of specific heats
    pres : float
        The pressure

    Returns
    -------
    out : float
       The internal energy density, rho e

    """
    rhoe = pres/(gamma - 1.0)
    return rhoe
