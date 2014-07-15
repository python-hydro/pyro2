#!/bin/sh

# this script builds each of the shared-object libraries that
# interface Fortran with python for some lower-level pyro routines.
# f2py is used.  

PYTHON=python

cd mesh
${PYTHON} setup.py build_ext --inplace
cd ..

cd incompressible
${PYTHON} setup.py build_ext --inplace
cd ..

cd compressible
${PYTHON} setup.py build_ext --inplace
cd ..

