#!/bin/bash

# this script builds the shared-object libraries that interface
# Fortran with python for some lower-level pyro routines.  f2py is
# used.
#
# use `./mk.sh clean` to clear all the build files
#
# set the python interpreter to use.  If no PYTHON variable is
# set, then default to python3.  You can use python2, for example,
# by doing:
# PYTHON=python2 ./mk.sh
: ${PYTHON:=python3}

if [ "$1" == "clean" ]; then

    find . -type d -name "__pycache__" -exec rm -rf {} \;
else

    ${PYTHON} setup.py build

fi
