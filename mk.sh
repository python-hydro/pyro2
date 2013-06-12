#!/bin/sh

# this script builds each of the shared-object libraries that
# interface Fortran with python for some lower-level pyro routines.
# f2py is used.  In each case, we simply 'cd' into the appropriate
# directory and do 'make'.  In case of a failure, you can try manually
# building the routine yourself with the 1-line f2py command in the
# makefile.

# reconstruction library
echo "making reconstruction_f.so..."
cd mesh
make >> /dev/null
err=$?
if [ $err -ne 0 ]; then
    echo "Error making reconstruction_f.so"
    exit $err
fi
cd ..

# compressible interface stuff
echo "making interface_f.so..."
cd compressible
make >> /dev/null
err=$?
if [ $err -ne 0 ]; then
    echo "Error making interface_f.so"
    exit $err
fi
cd ..

# incompressible interface stuff
echo "making incomp_interface_f.so..."
cd incompressible
make >> /dev/null
err=$?
if [ $err -ne 0 ]; then
    echo "Error making incomp_interface_f.so"
    exit $err
fi
cd ..

echo "done"
