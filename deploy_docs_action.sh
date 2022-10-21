#!/bin/bash
set -e # Exit with nonzero exit code if anything fails

mkdir out

# Build the Sphinx documentation
cd docs
make html
cd ../

mkdir -p out/
mv docs/build/html/* out/
