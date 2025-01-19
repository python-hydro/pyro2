*************************
Compressible source terms
*************************

Source terms are included in the compressible equations via a predictor-corrector formulation.  Starting
with the conserved state $\Uc^{n}$, we do the update as:

.. math::

   \begin{align*}
   \Uc^{n+1,\star} &= \Uc^n - \Delta t \left [ \nabla \cdot {\bf F}(\Uc)\right ]^{n+1/2} + \Delta t {\bf S}(\Uc^n) \\
   \Uc^{n+1} &= \Uc^{n+1,\star} + \frac{\Delta t}{2} ({\bf S}(\Uc^{n+1,\star}) - {\bf S}(\Uc^n))
   \end{align*}

Gravity
=======



Sponge
======



Heating term
============

