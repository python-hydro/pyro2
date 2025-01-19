*****************************
Compressible example problems
*****************************

Example problems
================

.. note::

   The 4th-order accurate solver (:py:mod:`pyro.compressible_fv4`) requires that
   the initialization create cell-averages accurate to 4th-order.  To
   allow for all the solvers to use the same problem setups, we assume
   that the initialization routines initialize cell-centers (which is
   fine for 2nd-order accuracy), and the
   :func:`preevolve() <pyro.compressible_fv4.simulation.Simulation.preevolve>` method will convert
   these to cell-averages automatically after initialization.


Sod
---

The Sod problem is a standard hydrodynamics problem. It is a
one-dimensional shock tube (two states separated by an interface),
that exhibits all three hydrodynamic waves: a shock, contact, and
rarefaction. Furthermore, there are exact solutions for a gamma-law
equation of state, so we can check our solution against these exact
solutions. See Toro's book for details on this problem and the exact
Riemann solver.

Because it is one-dimensional, we run it in narrow domains in the x- or y-directions. It can be run as:

.. prompt:: bash

   pyro_sim.py compressible sod inputs.sod.x
   pyro_sim.py compressible sod inputs.sod.y

A simple script, ``sod_compare.py`` in ``analysis/`` will read a pyro output
file and plot the solution over the exact Sod solution. Below we see
the result for a Sod run with 128 points in the x-direction, gamma =
1.4, and run until t = 0.2 s.

.. image:: sod_compare.png
   :align: center

We see excellent agreement for all quantities. The shock wave is very
steep, as expected. The contact wave is smeared out over ~5 zones—this
is discussed in the notes above, and can be improved in the PPM method
with contact steepening.

Sedov
-----

The Sedov blast wave problem is another standard test with an analytic
solution (Sedov 1959). A lot of energy is point into a point in a
uniform medium and a blast wave propagates outward. The Sedov problem
is run as:

.. prompt:: bash

   pyro_sim.py compressible sedov inputs.sedov

The video below shows the output from a 128 x 128 grid with the energy
put in a radius of 0.0125 surrounding the center of the domain. A
gamma-law EOS with gamma = 1.4 is used, and we run until 0.1

.. raw:: html

    <div style="position: relative; padding-bottom: 75%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/1JO6By78p9E?rel=0" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div><br>

We see some grid effects because it is hard to initialize a small
circular explosion on a rectangular grid. To compare to the analytic
solution, we need to radially bin the data. Since this is a 2-d
explosion, the physical geometry it represents is a cylindrical blast
wave, so we compare to Sedov's cylindrical solution. The radial
binning is done with the ``sedov_compare.py`` script in ``analysis/``

.. image:: sedov_compare.png
   :align: center

This shows good agreement with the analytic solution.


quad
----

The quad problem sets up different states in four regions of the
domain and watches the complex interfaces that develop as shocks
interact. This problem has appeared in several places (and a `detailed
investigation
<http://planets.utsc.utoronto.ca/~pawel/Riemann.hydro.html>`_ is
online by Pawel Artymowicz). It is run as:

.. prompt:: bash

   pyro_sim.py compressible quad inputs.quad

.. image:: quad.png
   :align: center


rt
--

The Rayleigh-Taylor problem puts a dense fluid over a lighter one and
perturbs the interface with a sinusoidal velocity. Hydrostatic
boundary conditions are used to ensure any initial pressure waves can
escape the domain. It is run as:

.. prompt:: bash

   pyro_sim.py compressible rt inputs.rt

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/P4zmObEYCOs?rel=0" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div><br>



bubble
------

The bubble problem initializes a hot spot in a stratified domain and
watches it buoyantly rise and roll up. This is run as:

.. prompt:: bash

   pyro_sim.py compressible bubble inputs.bubble


.. image:: bubble.png
   :align: center

The shock at the top of the domain is because we cut off the
stratified atmosphere at some low density and the resulting material
above that rains down on our atmosphere. Also note the acoustic signal
propagating outward from the bubble (visible in the U and e panels).
