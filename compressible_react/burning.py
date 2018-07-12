def energy_and_species_creation(cc_data, T):
    """
    Compute the heat release and species create rate
    """
    myg = cc_data.grid
    return myg.scratch_array(), myg.scratch_array()


def kappa(cc_data, temp, const, constant=1):
    """
    Conductivity

    If constant, just returns the constant defined in the params file.

    Otherwise, it uses the formula for conductivity with constant opacity in the Castro/Microphysics library.

    Parameters
    ----------
    cc_data : CellCenterData2d
        the cell centered data
    temp : ArrayIndexer
        temperature
    const : float
        the diffusion constant or opacity
    constant : int
        Is the conductivity constant (1) or the opacity constant (0)
    """
    myg = cc_data.grid
    k = myg.scratch_array()

    if constant == 1:
        k[:, :] = const
    else:
        sigma_SB = 5.6704e-5  # Stefan-Boltzmann in cgs
        dens = cc_data.get_var("density")
        k[:, :] = (16 * sigma_SB * temp**3) / (3 * const * dens)

    return k
