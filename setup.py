# this works for python 2 or 3 directly.  To build, do:
#
# python setup.py build_ext
#
# Note the setup.cfg directs the build to be done in-place.

from numpy.distutils.core import setup, Extension

ext_modules = [Extension("compressible.interface_f", ["compressible/interface_f.f90"]),
               Extension("advection_fv4.interface_f", ["advection_fv4/interface_states.f90"]),
               Extension("lm_atm.LM_atm_interface_f", ["lm_atm/LM_atm_interface_f.f90"]),
               Extension("incompressible.incomp_interface_f", ["incompressible/incomp_interface_f.f90"]),
               Extension("swe.interface_f", ["swe/interface_f.f90"])]

setup(ext_modules=ext_modules)
