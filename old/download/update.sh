#!/bin/sh

cd /home/www/hydro_by_example/download

cd _stage/pyro2
git pull
git log --name-only > ../../ChangeLog
cd ..

cd numerical_exercises
git pull
make

