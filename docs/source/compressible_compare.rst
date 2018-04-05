KH Problem with different solvers
*********************************
The McNally Kelvin-Helmholtz problem sets up a heavier fluid moving in the negative x-direction sandwiched between regions of lighter fluid moving in the positive x-direction.

The image below shows the KH problem initialized with McNally's test run with the different compressible solvers in pyro (standard Riemann, Runge-Kutta, fourth order). This is run with::

  ./pyro.py compressible kh inputs.kh mesh.nx=128 mesh.ny=128 kh.vbulk=0

  ./pyro.py compressible_rk kh inputs.kh mesh.nx=128 mesh.ny=128 kh.vbulk=0

  ./pyro.py compressible_fv4 kh inputs.kh mesh.nx=128 mesh.ny=128 kh.vbulk=0

.. image:: plot.png

We vary the velocity in the positive y-direction to see how effective the solvers are at preserving the initial shape.
