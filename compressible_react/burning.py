import numpy as np
import compressible.eos as eos
from scipy.integrate import odeint


class Network(object):
    """
    Class for holding info about the reaction network.
    """

    def __init__(self, nspec_evolve=0):
        """ Constructor """

        self.nspec = 0
        self.nspec_evolve = nspec_evolve
        self.naux = 0

        self.spec_names = []
        self.A_ion = []
        self.Z_ion = []
        self.E_binding = []

        self.nrates = 0

        self.do_constant_volume_burn = False
        self.self_heat = False
        self.call_eos_in_rhs = True
        self.dT_crit = 1e9

    def initialize(self, rp):
        try:
            self.Cv = rp.get_param("eos.cv")
        except KeyError:
            pass
        try:
            self.Cp = rp.get_param("eos.cp")
        except KeyError:
            pass

    def finalize(self):
        pass

    def enery_gen_rate(self, dydt):
        pass

    def temperature_rhs(self, cc_data, E_nuc):

        myg = cc_data.grid

        rhs = myg.scratch_array()

        if self.self_heat:
            if self.do_constant_volume_burn:
                rhs[:, :] = E_nuc / self.Cv

            else:  # constant pressure
                rhs[:, :] = E_nuc / self.Cp

        return rhs

    def energy_and_species_creation(self, cc_data):

        myg = cc_data.grid
        return myg.scratch_array(), myg.scratch_array(nvar=self.nspec_evolve)


class PowerLaw(Network):
    """
    powerlaw network. This is a single-step reaction rate. There are only two species, fuel f and ash a which react through the reaction $f + f \rightarrow a + \gamma$. Baryon conservation requires that $A_f = A_a/2$ and charge conservation requires that $Z_f = Z_a/2$. The reaction rate is a powerlaw in temperature.
    """

    def __init__(self):
        """ Constructor """
        super().__init__(nspec_evolve=2)
        self.nspec = 3
        self.self_heat = True

    def initialize(self, rp):
        super().initialize(rp)

        self.spec_names = ["fuel", "ash", "inert"]

        self.A_ion = np.array([2, 4, 8])
        self.Z_ion = np.array([1, 2, 4])
        self.E_binding = np.array(
            [0, rp.get_param("network.specific_q_burn"), 0])

        self.F_act = rp.get_param("network.f_act")
        self.T_burn_ref = rp.get_param("network.t_burn_ref")
        self.rho_burn_ref = rp.get_param("network.rho_burn_ref")
        self.rtilde = rp.get_param("network.rtilde")
        self.nu = rp.get_param("network.nu")


    def enery_gen_rate(self, dydt):
        """
        Computers instantaneous energy generation rate
        """
        return np.sum(dydt[:, :, :] *
                      self.A_ion[np.newaxis,
                                 np.newaxis, :self.nspec_evolve] *
                      self.E_binding[np.newaxis,
                                     np.newaxis, :self.nspec_evolve],
                      axis=2)

    def energy_and_species_creation(self, cc_data):
        """
        RHS of reactions ODE
        """

        myg = cc_data.grid

        xfueltmp = np.maximum(cc_data.get_var("fuel"), 0)
        dens = cc_data.get_var("density")
        eint = cc_data.get_var("eint")

        T = myg.scratch_array()
        T[:, :] = eos.temp(eint, self.Cv)

        rate = myg.scratch_array()
        omega_dot = myg.scratch_array(nvar=self.nspec_evolve)

        mask = (T >= self.F_act * self.T_burn_ref)

        rate[mask] = self.rtilde * dens[mask] / self.rho_burn_ref * \
            xfueltmp[mask]**2 * (T[mask] / self.T_burn_ref)**self.nu

        omega_dot[:, :, 0] = -rate
        omega_dot[:, :, 1] = rate

        omega_dot[:, :, :] /= \
            self.A_ion[np.newaxis,
                       np.newaxis, :self.nspec_evolve]

        E_nuc = self.enery_gen_rate(omega_dot)

        # net_T = self.temperature_rhs(cc_data, E_nuc)

        return E_nuc, omega_dot


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
