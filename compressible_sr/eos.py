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
    return dens*eint*(gamma - 1.0)


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
    return pres/(eint*(gamma - 1.0))


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
    return pres/(gamma - 1.0)


def h_from_eps(gamma, eint):
    """
    Given rho and internal energy, return h
    """

    return 1 + gamma * eint


def rhoh_from_rho_p(gamma, rho, p):
    """
    Given rho and p, return h
    """
    return rho + gamma / (gamma - 1) * p
