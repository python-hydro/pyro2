import numpy as np
import compressible.eos as eos


class Network(object):
    """
    Implements calculations to do with the reaction network.

    This is a base class which implements the null case (no burning). Other networks should inherit from this one.
    """

    def __init__(self, nspec_evolve=0):
        """
        Constructor

        Parameters
        ----------
        nspec_evolve : int
            Number of species to evolve.
        """

        self.nspec = 0
        self.nspec_evolve = nspec_evolve
        self.naux = 0

        self.spec_names = None
        self.A_ion = []
        self.Z_ion = []
        self.E_binding = []

        self.do_constant_volume_burn = False
        self.self_heat = False
        self.call_eos_in_rhs = True
        self.dT_crit = 1e9

    def initialize(self, rp):
        """
        Initialize the class. Will try to find the heat capacities at constant volume and pressure.
        """
        try:
            self.Cv = rp.get_param("eos.cv")
        except KeyError:
            pass
        try:
            self.Cp = rp.get_param("eos.cp")
        except KeyError:
            pass

    def temperature_rhs(self, cc_data, E_nuc):
        """
        Calculates the time derivative of the temperature.

        Parameters
        ----------
        cc_data : CellCenterData2d
            Cell-centered data
        E_nuc : Grid2d
            Instantaneous energy generation rate.
        """

        myg = cc_data.grid

        rhs = myg.scratch_array()

        if self.self_heat:
            if self.do_constant_volume_burn:
                rhs[:, :] = E_nuc / self.Cv

            else:  # constant pressure
                rhs[:, :] = E_nuc / self.Cp

        return rhs

    def energy_and_species_creation(self, cc_data):
        """
        Returns the instantaneous energy generation rate and the species creation rate.

        Parameters
        ----------
        cc_data : CellCenterData2d
            The cell-centered data
        """

        myg = cc_data.grid
        return myg.scratch_array(), myg.scratch_array(nvar=self.nspec_evolve)


class PowerLaw(Network):
    r"""
    powerlaw network. This is a single-step reaction rate.
    There are only two species, fuel :math:`f` and ash :math:`a` which react
    through the reaction :math:`f + f \rightarrow a + \gamma`.
    Baryon conservation requires that :math:`A_f = A_a/2` and
    charge conservation requires that :math:`Z_f = Z_a/2`. The
    reaction rate is a powerlaw in temperature.
    """

    def __init__(self):
        """ Constructor """
        super().__init__(nspec_evolve=2)
        self.nspec = 3
        self.self_heat = True

    def initialize(self, rp):
        """
        Initialize the object, loading up a number of runtime parameters.

        Parameters
        ----------
        rp : RuntimeParameters
            Runtime paramters
        """
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
        r"""
        Computes instantaneous energy generation rate. It is given by

        .. math::

            f(\dot{Y}_k) = \sum_k \dot{Y}_k A_k E_{\text{bin},k},

        where :math:`E_{\text{bin},k}` is the binding energy of species :math:`k`.

        Parameters
        ----------
        dydt : Grid2d
            Species creation rate
        """
        return np.sum(dydt[:, :, :] *
                      self.A_ion[np.newaxis,
                                 np.newaxis, :self.nspec_evolve] *
                      self.E_binding[np.newaxis,
                                     np.newaxis, :self.nspec_evolve],
                      axis=2)

    def energy_and_species_creation(self, cc_data):
        r"""
        Returns the instantaneous energy generation rate and the species creation rate.

        The rate is given by

        .. math::

            \dot{\omega}_k = \tilde{r} \frac{\rho}{\rho_{\text{ref}}}X_k^2 \left(\frac{T}{T_{\text{ref}}}\right)^\nu,

        where :math:`\tilde{r}` is the coefficient for the reaction rate, :math:`\rho_{\text{ref}}` and :math:`T_{\text{ref}}` are the reference density and temperature, and :math:`\nu` is the exponent for the temperature.

        :math:`\tilde{r}` is zero if the temperature is below some activation temperature, given by some fraction :math:`F_{\text{act}}` of the reference temperature.

        Parameters
        ----------
        cc_data : CellCenterData2d
            The cell-centered data
        """

        myg = cc_data.grid

        dens = cc_data.get_var("density")
        eint = cc_data.get_var("eint")
        xfueltmp = np.maximum(cc_data.get_var("fuel")/dens, 0)

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


def k_th(cc_data, temp, const, constant=1):
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
