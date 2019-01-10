import numpy as np

import compressible.eos as eos


def derive_primitives(myd, varnames):
    """
    derive desired primitive variables from conserved state
    """

    # get the variables we need
    dens = myd.get_var("density")
    xmom = myd.get_var("x-momentum")
    ymom = myd.get_var("y-momentum")
    ener = myd.get_var("energy")

    bx = myd.get_var("x-magnetic-field")
    by = myd.get_var("y-magnetic-field")

    derived_vars = []

    u = xmom/dens
    v = ymom/dens

    e = (ener - 0.5*dens*(u*u + v*v))/dens

    gamma = myd.get_aux("gamma")
    p = eos.pres(gamma, dens, e)

    if isinstance(varnames, str):
        wanted = [varnames]
    else:
        wanted = list(varnames)

    for var in wanted:

        if var == "velocity":
            derived_vars.append(u)
            derived_vars.append(v)

        elif var in ["e", "eint"]:
            derived_vars.append(e)

        elif var in ["p", "pressure"]:
            derived_vars.append(p)

        elif var == "primitive":
            derived_vars.append(dens)
            derived_vars.append(u)
            derived_vars.append(v)
            derived_vars.append(p)

        elif var == "soundspeed":
            derived_vars.append(np.sqrt(gamma*p/dens))

        elif var == "alfven":
            derived_vars.append(np.sqrt((bx**2 + by**2) / dens))

        elif var == "x-magnetosonic":
            c2 = gamma*p/dens
            cA2 = (bx**2 + by**2) / dens
            cAx2 = bx**2 / dens
            cf = np.sqrt(
                0.5 * (c2 + cA2 + np.sqrt((c2 + cA2)**2 - 4 * c2 * cAx2)))
            cs = cf = np.sqrt(
                0.5 * (c2 + cA2 - np.sqrt((c2 + cA2)**2 - 4 * c2 * cAx2)))
            derived_vars.append(cf)
            derived_vars.append(cs)

        elif var == "y-magnetosonic":
            c2 = gamma*p/dens
            cA2 = (bx**2 + by**2) / dens
            cAy2 = by**2 / dens
            cf = np.sqrt(
                0.5 * (c2 + cA2 + np.sqrt((c2 + cA2)**2 - 4 * c2 * cAy2)))
            cs = cf = np.sqrt(
                0.5 * (c2 + cA2 - np.sqrt((c2 + cA2)**2 - 4 * c2 * cAy2)))
            derived_vars.append(cf)
            derived_vars.append(cs)

    if len(derived_vars) > 1:
        return derived_vars
    else:
        return derived_vars[0]
