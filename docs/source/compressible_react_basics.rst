Reactive flow
=============

The ``compressible_react`` solver can be used for modelling compressible hydrodynamics with reactions.

The equations of compressible hydrodynamics with reactive and diffusive source terms take the form:

.. math::

    \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho U) &= 0 \\
    \frac{\partial \left(\rho X_k\right)}{\partial t} + \nabla \cdot (\rho U X_k) &= \rho \dot{\omega}_k \\
    \frac{\partial (\rho U)}{\partial t} + \nabla \cdot (\rho U U) + \nabla p &= \rho g \\
    \frac{\partial (\rho E)}{\partial t} + \nabla \cdot [(\rho E + p ) U] &= \nabla \cdot \left(k_{\text{th}} \nabla T\right) + \rho H_{\text{nuc}}

where :math:`X_k = \rho_k/\rho` is the mass fraction of species :math:`k` (and :math:`\sum_k X_k = 1`), :math:`\dot{\omega}_k` is the species creation rate, :math:`k_{\text{th}}` is the thermal conductivity, :math:`T` is the temperature and :math:`H_{\text{nuc}}` is the time-rate of energy release per unit mass.

Burning and diffusion are implemented using *Strang splitting*. We evolve the burning and diffusion terms by half a timestep, evolve the compressible hydrodynamics terms by a full timestep, then evolve the diffusion and burning terms by the remaining half timestep. This approach makes the operator splitting second-order in time.


Burning
-------
In order include reactions in our simulation, we need to define a reaction network. This is handled by the :func:`Network <compressible_react.burning.Network>` class. This class allows you to define the number of species in the network, the species' properties (atomic mass, proton number and binding energy), and how they react. Once the network has been initialized at the beginning of the simulation, the instantaneous energy generation rate and the species creation rate can be found by calling the member function :func:`energy_and_species_creation <compressible_react.burning.Network.energy_and_species_creation>`.

The base ``Network`` class implements a 'null' network - no burning or species. A single-step reaction network is implemented in the :func:`PowerLaw <compressible_react.burning.PowerLaw>` class. This models a network made up of two species: fuel, :math:`f`, and ash, :math:`a`. They react
through the reaction

.. math::
    f + f \rightarrow a + \gamma.

Baryon conservation requires that :math:`A_f = A_a/2` and
charge conservation requires that :math:`Z_f = Z_a/2`. The
reaction rate is a powerlaw in temperature:

.. math::

    \dot{\omega}_k = \tilde{r} \frac{\rho}{\rho_{\text{ref}}}X_k^2 \left(\frac{T}{T_{\text{ref}}}\right)^\nu,

where :math:`\tilde{r}` is the coefficient for the reaction rate, :math:`\rho_{\text{ref}}` and :math:`T_{\text{ref}}` are the reference density and temperature, and :math:`\nu` is the exponent for the temperature.

:math:`\tilde{r}` is zero if the temperature is below some activation temperature, given by some fraction :math:`F_{\text{act}}` of the reference temperature.

The reaction network can be configured using the following runtime parameters:

+--------------------------------------------------------------------------------+
|``[network]``                                                                   |
+=======================+========================================================+
|``network_type``       | What type of network? ``null`` or ``powerlaw``         |
+-----------------------+--------------------------------------------------------+
|``f_act``              | activation temperature factor                          |
+-----------------------+--------------------------------------------------------+
|``t_burn_ref``         | reference temperature                                  |
+-----------------------+--------------------------------------------------------+
|``rho_burn_ref``       | reference density                                      |
+-----------------------+--------------------------------------------------------+
|``rtilde``             | coefficient for the reaction rate                      |
+-----------------------+--------------------------------------------------------+
|``nu``                 | exponent for the temperature                           |
+-----------------------+--------------------------------------------------------+



Diffusion
---------

Diffusion follows Fick's law: the heat flux is proportional to the gradient of the temperature, :math:`F_{\text{cond}} = -k_{\text{th}}\nabla T`. The thermal conductivity :math:`-k_{\text{th}}` can be calculated in a number of ways. The simplest is to set it to be constant everywhere. Alternatively, we can set the opacity :math:`\kappa` to be constant. In this case, the conductivity is given by

.. math::

    k_{\text{th}} = \frac{16 \sigma_{\text{SB}} T^3}{3 \kappa \rho},

where :math:`\sigma_{\text{SB}}` is the Stefan-Boltzmann constant.

The diffusion can be configured using the following runtime parameters:

+--------------------------------------------------------------------------------+
|``[diffusion]``                                                                 |
+=======================+========================================================+
|``k``                  | conductivity constant                                  |
+-----------------------+--------------------------------------------------------+
|``constant_kappa``     | Constant conductivity (1) or opacity (0)?              |
+-----------------------+--------------------------------------------------------+


Code examples
-------------

To include reactions in the code, we need to first create and initialize the reaction network. We do this by creating a ``Network`` object (or an object of a subclass), then calling the object's :func:`initialize <compressible_react.burning.Network.initialize>` function. This is done in the ``Simulation.initialize`` function in ``compressible_react``.

.. code-block:: python

    from util import runparams
    import compressible_react.burning as burning

    # create and import runtime parameters from file
    rp = runparams.RuntimeParameters()

    ....

    network = burning.PowerLaw()
    network.initialize(rp)

The internal energy and conserved species mass fractions are updated by calling the :func:`burn <compressible_react.Simulation.burn>` function. This in turn calls the network object's :func:`energy_and_species_creation <compressible_react.burning.Network.energy_and_species_creation>` function to calculate the instantaneous energy generation rate and species creation rates:

.. code-block:: python

    H, omega_dot = network.energy_and_species_creation(cc_data)

where ``cc_data`` is a ``CellCenterData2d`` object.

The diffusion term in the energy equation is found using :func:`diffusion <compressible_react.Simulation.diffusion>`. This calls the :func:`k_th <compressible_react.burning.k_th>` function to calculate the conductivity:

.. code-block:: python

    k = burning.k_th(cc_data, temp, k_const),

then computes :math:`\nabla\cdot (k\nabla T)`.
