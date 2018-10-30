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

    rm -rf mesh/*.so
    rm -rf incompressible/*.so
    rm -rf compressible/*.so
    rm -rf lm_atm/*.so
    find . -name "*.pyc" -exec rm -f {} \;
    find . -type d -name "__pycache__" -exec rm -rf {} \;
    find . -type d -name "build" -exec rm -rf {} \;

elif [ "$1" == "fortran" ]; then

    if [ "$1" == "debug" ]; then
	FFLAGS="-fbounds-check -fbacktrace -Wuninitialized -Wunused -ffpe-trap=invalid -finit-real=snan"
    else
	FFLAGS="-C"
    fi

    ${PYTHON} setup_fortran.py config_fc --f90flags "${FFLAGS}" build_ext

else

    ${PYTHON} setup.py build

fi
