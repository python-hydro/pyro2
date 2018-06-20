import numpy as np

import compressible_sr.eos as eos
import compressible_sr as comp


def derive_primitives(myd, varnames, ivars, myg):
    """
    derive desired primitive variables from conserved state
    """

    # get the variables we need
    # dens = myd.get_var("densityW")
    # xmom = myd.get_var("x-momentum")
    # ymom = myd.get_var("y-momentum")
    # ener = myd.get_var("energy")

    gamma = myd.get_aux("gamma")

    q = comp.cons_to_prim(myd.data, gamma, ivars, myg)

    derived_vars = []

    dens = q[:, :, ivars.irho]
    u = q[:, :, ivars.iu]
    v = q[:, :, ivars.iv]
    p = q[:, :, ivars.ip]
    e = eos.rhoe(gamma, p)/dens

    # u = xmom/dens
    # v = ymom/dens
    #
    # e = (ener - 0.5*dens*(u*u + v*v))/dens

    gamma = myd.get_aux("gamma")
    # p = eos.pres(gamma, dens, e)

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

    if len(derived_vars) > 1:
        return derived_vars
    else:
        return derived_vars[0]
