# this works for python 2 or 3 directly.  To build, do:
#
# python setup.py build_ext --inplace

from numpy.distutils.core import setup, Extension

extra_link_args=[]

ext = Extension("incomp_interface_f", 
                ["incomp_interface_f.f90"])

setup(ext_modules=[ext])

                
