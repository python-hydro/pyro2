"""
This is a gamma-law equation of state: p = rho e (gamma - 1), where
gamma is the constant ratio of specific heats.
"""


def pres(gamma, rho, eint):
    """
    Given the density and the specific internal energy, return the
    pressure

    Parameters
    ----------
    gamma : float
        The ratio of specific heats
    rho : float
        The density
    eint : float
        The specific internal energy

    Returns
    -------
    out : float
       The pressure

     """
    p = rho * eint * (gamma - 1.0)
    return p


def dens(gamma, p, eint):
    """
    Given the pressure and the specific internal energy, return
    the density

    Parameters
    ----------
    gamma : float
        The ratio of specific heats
    p : float
        The pressure
    eint : float
        The specific internal energy

    Returns
    -------
    out : float
       The density

    """
    rho = p / (eint * (gamma - 1.0))
    return rho


def rhoe(gamma, p):
    """
    Given the pressure, return (rho * e)

    Parameters
    ----------
    gamma : float
        The ratio of specific heats
    p : float
        The pressure

    Returns
    -------
    out : float
       The internal energy density, rho e

    """
    return p / (gamma - 1.0)
