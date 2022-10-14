import numpy as np

import pyro.compressible_sr.eos as eos
import pyro.compressible_sr.unsplit_fluxes as flx


def derive_primitives(myd, varnames, ivars, myg):
    """
    derive desired primitive variables from conserved state
    """

    # get the variables we need
    gamma = myd.get_aux("gamma")

    q = flx.cons_to_prim_wrapper(myd.data, gamma, ivars, myg)

    derived_vars = []

    dens = q[:, :, ivars.irho]
    u = q[:, :, ivars.iu]
    v = q[:, :, ivars.iv]
    p = q[:, :, ivars.ip]
    try:
        e = eos.rhoe(gamma, p)/dens
    except FloatingPointError:
        p[:, :] = myd.data[:, :, ivars.iener] * (gamma-1)
        e = myd.data[:, :, ivars.iener]  # p / (gamma - 1)

    gamma = myd.get_aux("gamma")
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
