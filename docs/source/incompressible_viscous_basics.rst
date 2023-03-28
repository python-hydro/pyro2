Incompressible hydrodynamics solver
===================================

pyro's incompressible solver solves:

.. math::

   \frac{\partial U}{\partial t} + U \cdot \nabla U + \nabla p &= 0 \\
   \nabla \cdot U &= 0

The algorithm combines the Godunov/advection features
used in the advection and compressible solver together with multigrid
to enforce the divergence constraint on the velocities.

Here we implement a cell-centered approximate projection method for
solving the incompressible equations. At the moment, only periodic BCs
are supported.

The main parameters that affect this solver are:

.. include:: incompressible_viscous_defaults.inc

Examples
--------

cavity

The lid-driven cavity is a well-known benchmark problem for hydro codes
(see e.g. )
^^^^^

The shear problem initializes a shear layer in a domain with
doubly-periodic boundaries and looks at the development of two
vortices as the shear layer rolls up. This problem was explored in a
number of papers, for example, Bell, Colella, & Glaz (1989) and Martin
& Colella (2000). This is run as:

.. prompt:: bash

   pyro_sim.py incompressible shear inputs.shear


.. image:: shear.png
   :align: center

The vorticity panel (lower left) is what is usually shown in
papers. Note that the velocity divergence is not zero—this is because
we are using an approximate projection.

convergence
^^^^^^^^^^^

The convergence test initializes a simple velocity field on a periodic
unit square with known analytic solution. By evolving at a variety of
resolutions and comparing to the analytic solution, we can measure the
convergence rate of the algorithm. The particular set of initial
conditions is from Minion (1996). Limiting can be disabled by adding
``incompressible.limiter=0`` to the run command. The basic set of tests
shown below are run as:

.. prompt:: bash

   pyro_sim.py incompressible converge inputs.converge.32 vis.dovis=0
   pyro_sim.py incompressible converge inputs.converge.64 vis.dovis=0
   pyro_sim.py incompressible converge inputs.converge.128 vis.dovis=0

The error is measured by comparing with the analytic solution using
the routine ``incomp_converge_error.py`` in ``analysis/``.

.. image:: incomp_converge.png
   :align: center

The dashed line is second order convergence. We see almost second
order behavior with the limiters enabled and slightly better than
second order with no limiting.

Exercises
---------

Suggestions welcome!