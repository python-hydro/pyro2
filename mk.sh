#!/bin/sh

# this script builds each of the shared-object libraries that
# interface Fortran with python for some lower-level pyro routines.
# f2py is used.  In each case, we simply 'cd' into the appropriate
# directory and do 'make'.  In case of a failure, you can try manually
# building the routine yourself with the 1-line f2py command in the
# makefile.

function build_lib  # args: directory libname
{
  echo "making $2"
  cd $1
  make >> /dev/null
  err=$?
  if [ $err -ne 0 ]; then
    echo "Error making $2"
    exit $err
  fi
  cd ..
}

build_lib mesh reconstruction_f.so
build_lib compressible interface_f.so
build_lib  incompressible incomp_interface_f.so

echo "done"
