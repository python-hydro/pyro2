gamma = None

def init(gamma_constant):
    global gamma
    gamma = gamma_constant


def pres(dens, eint):
    """ given the density and the specific internal energy, return the
        pressure """

    global gamma
    p = dens*eint*(gamma - 1.0)

    return p


def dens(pres, eint):
    """ given the pressure and the specific internal energy, return
        the density """

    global gamma
    dens = pres/(eint*(gamma - 1.0))

    return dens


def rhoe(pres):
    """ given the pressure, return (rho * e) """

    global gamma
    rhoe = pres/(gamma - 1.0)

    return rhoe


