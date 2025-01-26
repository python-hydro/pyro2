History
=======

The original pyro code was written in 2003-4 to help developer
Zingale understand these methods for himself. It was originally written
using the Numeric array package and handwritten C extensions for the
compute-intensive kernels.  It was ported to numarray when that
replaced Numeric, and continued to use C extensions.  This version
"pyro2" was resurrected beginning in 2012 and rewritten for numpy
using f2py, and brought up to date.  Most recently we've dropped
f2py and are using numba for the compute-intensive kernels.

