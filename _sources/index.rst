.. pyro documentation main file, created by
   sphinx-quickstart on Mon Dec 25 18:42:54 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*************************
pyro: a python hydro code
*************************

`https://github.com/python-hydro/pyro2 <https://github.com/python-hydro/pyro2>`_

.. image:: pyro_plots.png

About
=====

pyro is a python hydrodynamics code meant to illustrate how the basic methods
used in astrophysical simulations work.  It is also used for prototyping
new ideas.

.. toctree::
   :maxdepth: 1
   :caption: pyro basics
   :hidden:

   intro
   installation
   notes
   design
   running
   output
   problems

.. toctree::
   :maxdepth: 1
   :caption: Mesh
   :hidden:

   mesh_basics
   mesh-examples.ipynb
   spherical-mesh.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Multigrid
   :hidden:

   multigrid
   multigrid_basics
   multigrid-constant-coefficients
   multigrid-variable-coeff
   multigrid-general-linear

.. toctree::
   :maxdepth: 1
   :caption: Hydro Solvers
   :hidden:

   advection_basics
   burgers_basics
   compressible_basics
   diffusion_basics
   incompressible_basics
   incompressible_viscous_basics
   lowmach_basics
   swe_basics
   particles_basics

.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   compressible-rt-compare.ipynb
   adding_a_problem_jupyter.ipynb
   advection-error.ipynb
   compressible-convergence.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Utilities
   :hidden:

   analysis
   testing

.. toctree::
   :maxdepth: 1
   :caption: Community
   :hidden:

   help
   ack

.. toctree::
   :maxdepth: 1
   :caption: Software Reference
   :hidden:

   API <modules>

.. toctree::
   :caption: Bibliography
   :hidden:

   zreferences


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
