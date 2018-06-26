import numpy as np


def derive_primitives(myd, varnames):
    """
    derive desired primitive variables from conserved state
    """

    # get the variables we need
    h = myd.get_var("height")
    xmom = myd.get_var("x-momentum")
    ymom = myd.get_var("y-momentum")

    derived_vars = []

    u = xmom/h
    v = ymom/h

    g = myd.get_aux("g")

    if isinstance(varnames, str):
        wanted = [varnames]
    else:
        wanted = list(varnames)

    for var in wanted:

        if var == "velocity":
            derived_vars.append(u)
            derived_vars.append(v)

        elif var == "primitive":
            derived_vars.append(h)
            derived_vars.append(u)
            derived_vars.append(v)

        elif var == "soundspeed":
            derived_vars.append(np.sqrt(g*h))

    if len(derived_vars) > 1:
        return derived_vars
    else:
        return derived_vars[0]
