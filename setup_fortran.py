# this works for python 2 or 3 directly.  To build, do:
#
# python setup.py build_ext
#
# Note the setup.cfg directs the build to be done in-place.

from numpy.distutils.core import setup, Extension

f_modules = ["compressible.interface", "advection_fv4.interface",
             "lm_atm.LM_atm_interface", "incompressible.incomp_interface", "swe.interface"]

ext_modules = [Extension(module, [module.replace('.', '/') + '_f.f90'])
               for module in f_modules]

setup(ext_modules=ext_modules)
