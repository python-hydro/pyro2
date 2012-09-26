#!/bin/sh

# reconstruction library
echo "making reconstruction_f.so..."
cd mesh
make
cd ..

echo "making interface_f.so..."
cd compressible
make
cd ..

echo "done"
