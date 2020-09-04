# this works for python 2 or 3 directly.  To build, do:
#
# python setup.py build_ext
#
# Note the setup.cfg directs the build to be done in-place.

from setuptools import setup

setup(name='pyro',
      version='2.2.0',
      url='https://github.com/python-hydro/pyro2',
      license='BSD',
      install_requires=['numpy', 'numba', 'matplotlib', 'h5py'])

