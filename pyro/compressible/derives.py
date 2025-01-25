import numpy as np

from pyro.compressible import eos


def derive_primitives(myd, varnames):
    """
    derive desired primitive variables from conserved state
    """

    # get the variables we need
    dens = myd.get_var("density")
    xmom = myd.get_var("x-momentum")
    ymom = myd.get_var("y-momentum")
    ener = myd.get_var("energy")

    derived_vars = []

    u = xmom/dens
    v = ymom/dens

    e = (ener - 0.5*dens*(u*u + v*v))/dens

    gamma = myd.get_aux("gamma")
    p = eos.pres(gamma, dens, e)

    myg = myd.grid
    vort = myg.scratch_array()

    vort.v()[:, :] = \
        0.5*(v.ip(1) - v.ip(-1))/myg.dx - \
        0.5*(u.jp(1) - u.jp(-1))/myg.dy

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

        elif var == "machnumber":
            derived_vars.append(np.sqrt(u**2 + v**2) / np.sqrt(gamma*p/dens))

        elif var == "vorticity":
            derived_vars.append(vort)

    if len(derived_vars) > 1:
        return derived_vars

    return derived_vars[0]
