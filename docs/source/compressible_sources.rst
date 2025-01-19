*************************
Compressible source terms
*************************

Implementation
==============

Various source terms are included in the compressible equation set.  Their implementation is solver-dependent.

``compressible``
----------------

For the ``compressible`` solver, source terms are included in the
compressible equations via a predictor-corrector formulation.
Starting with the conserved state, $\Uc^{n}$, we do the update as:

.. math::

   \begin{align*}
   \Uc^{n+1,\star} &= \Uc^n - \Delta t \left [ \nabla \cdot {\bf F}(\Uc)\right ]^{n+1/2} + \Delta t {\bf S}(\Uc^n) \\
   \Uc^{n+1} &= \Uc^{n+1,\star} + \frac{\Delta t}{2} ({\bf S}(\Uc^{n+1,\star}) - {\bf S}(\Uc^n))
   \end{align*}

This time-centers the sources, making the update second-order accurate.

Gravity
=======



Sponge
======



Heating term
============

