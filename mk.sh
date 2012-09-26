#!/bin/sh

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

echo "done"
