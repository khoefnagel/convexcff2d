#!/bin/bash

for subdir in c*/; do
	cd "$subdir"
	./clean.sh
	cd ../
done

cd OpenFOAMCoarse/
./clean.sh
cd ..

cd OpenFOAMFine/
./clean.sh
cd ..

rm -f *.pdf
