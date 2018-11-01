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

    rm -rf advection_fv4/*.so
    rm -rf compressible/*.so
    rm -rf incompressible/*.so
    rm -rf lm_atm/*.so
    rm -rf mesh/*.so
    rm -rf swe/*.so
    find . -name "*.pyc" -exec rm -f {} \;
    find . -type d -name "__pycache__" -exec rm -rf {} \;
    find . -type d -name "build" -exec rm -rf {} \;

    # move numba interface files back
    regex="([a-z0-9A-Z_/]*/)_([a-z0-9A-Z_]*interface.py)"
    for f in $(find . -name "*interface.py")
    do
        if [[ $f =~ $regex ]]
        then
            mv ".${BASH_REMATCH[1]}_${BASH_REMATCH[2]}" ".${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
        fi
    done

elif [ "$1" == "fortran" ]; then

    if [ "$1" == "debug" ]; then
	FFLAGS="-fbounds-check -fbacktrace -Wuninitialized -Wunused -ffpe-trap=invalid -finit-real=snan"
    else
	FFLAGS="-C"
    fi

    # move numba interface files out of the way
    regex="([a-z0-9A-Z_/]*/)([a-z0-9A-Z_]*interface.py)"
    for f in $(find . -name "*interface.py")
    do
        if [[ $f =~ $regex ]]
        then
            mv ".${BASH_REMATCH[1]}${BASH_REMATCH[2]}" ".${BASH_REMATCH[1]}_${BASH_REMATCH[2]}"
        fi
    done

    ${PYTHON} setup_fortran.py config_fc --f90flags "${FFLAGS}" build_ext

else

    ${PYTHON} setup.py build

fi
