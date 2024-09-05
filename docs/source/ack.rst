Acknowledgments
===============

The current pyro developers are listed in the `.zenodo.json` file that
is used for releases.

You are free to use this code and the accompanying notes in your
classes. Please credit "pyro development team" for the code, and
*please send a note to the pyro-help e-mail list describing how you
use it, so we can keep track of it (and help justify the development
effort).*

If you use pyro in a publication, please cite it using this bibtex
citation::

    @article{pyro,
      doi = {10.21105/joss.01265},
      url = {https://doi.org/10.21105/joss.01265},
      year = {2019},
      publisher = {The Open Journal},
      volume = {4},
      number = {34},
      pages = {1265},
      author = {Alice Harpole and Michael Zingale and Ian Hawke and Taher Chegini},
      title = {pyro: a framework for hydrodynamics explorations and prototyping},
      journal = {Journal of Open Source Software}
    }

pyro benefited from numerous useful discussions with Ann Almgren, John
Bell, and Andy Nonaka.


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

