#!/bin/sh

# this script builds each of the shared-object libraries that
# interface Fortran with python for some lower-level pyro routines.
# f2py is used.  

# set the python interpreter to use.  If no PYTHON variable is 
# set, then default to python.  You can use python2, for example,
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
    
else
    if [ "$1" == "debug" ]; then
	FFLAGS="-fbounds-check -fbacktrace -Wuninitialized -Wunused -ffpe-trap=invalid -finit-real=snan"
    else
	FFLAGS="-C"
    fi
    
    for d in mesh incompressible compressible lm_atm
    do
	cd ${d}
	${PYTHON} setup.py config_fc --f90flags "${FFLAGS}" build_ext --inplace
	cd ..
    done
fi

