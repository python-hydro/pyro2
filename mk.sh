#!/bin/sh

# this script builds each of the shared-object libraries that
# interface Fortran with python for some lower-level pyro routines.
# f2py is used.  

# set the python interpreter to use.  If no PYTHON variable is 
# set, then default to python.  You can use python3, for example,
# by doing:
# PYTHON=python3 ./mk.sh
: ${PYTHON:=python}

cd mesh
${PYTHON} setup.py build_ext --inplace
cd ..

cd incompressible
${PYTHON} setup.py build_ext --inplace
cd ..

cd compressible
${PYTHON} setup.py build_ext --inplace
cd ..

cd lm_atm
${PYTHON} setup.py build_ext --inplace
cd ..

