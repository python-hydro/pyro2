Particles overview
==================

A solver for modelling particles.

``particles.particles`` implementation and use
----------------------------------------------

We import the basic particles functionality as:

.. code-block:: python

   import particles.particles as particles

The particles solver is made up of two classes:

* :func:`Particle <particles.particles.Particle>`, which holds
  the data about a single particle (its position and velocity);
* :func:`Particles <particles.particles.Particles>`, which holds the data
  about a collection of particles.

The particles are stored as a dictionary, and their positions are updated
based on the velocity on the grid. The keys are assumed to be the initial
positions of the particles (however this is only used for plotting purposes
so any tuple would be fine).

The particles can be initialised in a number of ways:

* :func:`randomly_generate_particles <particles.particles.Particle.randomly_generate_particles>`,
  which randomly generates ``n_particles`` within the domain;
* :func:`grid_generate_particles <particles.particles.Particle.grid_generate_particles>`,
  which will generate approximately ``n_particles`` equally spaced in the
  x-direction and y-direction (though the uses the same number of particles in
  each direction, so the spacing will be different in each direction if the
  domain is not square.) The number of particles will be increased/decreased
  in order to fill the whole domain;
* the user can define their own ``particle_generator_func`` and pass this into the
  Particles constructor. This function takes the number of particles to be
  generated and returns a dictionary of ``Particle`` objects.

We can initialize particles in a problem using the following code in the
solver's ``Simulation.initialize`` function:

.. code-block:: python

   if self.rp.get_param("particles.do_particles") == 1:
       self.particles = particles.Particles(self.cc_data, bc, self.rp)

The particles can then be advanced by inserting the following code after the
update of the other variables in the solver's ``Simulation.evolve`` function:

.. code-block:: python

   if self.particles is not None:
        self.particles.update_particles(u, v, self.dt)
        self.particles.enforce_particle_boundaries()

where ``u`` and ``v`` are the ``ArrayIndexer`` objects holding the x-velocity and
y-velocity on the grid.

We can turn on/off the particles solver using the following runtime paramters:

+--------------------------------------------------------------------------------+
|``[particles]``                                                                 |
+=======================+========================================================+
|``do_particles``       | do we want to model particles? (0=no, 1=yes)           |
+-----------------------+--------------------------------------------------------+
|``n_particles``        | number of particles to be modelled                     |
+-----------------------+--------------------------------------------------------+
|``particle_generator`` | how do we initialize the particles? "random"           |
|                       | randomly generates particles throughout the domain,    |
|                       | "grid" generates equally spaced particles. This        |
|                       | option can be overridden by passing a custom generator |
|                       | function to the ``Particles`` constructor.             |
+-----------------------+--------------------------------------------------------+
