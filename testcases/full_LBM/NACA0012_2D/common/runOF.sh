#!/bin/bash

python3 datToGeo.py
gmsh -3 setupCase.geo -format msh2 -o output.msh > log.gmsh


gmshToFoam output.msh > log.gmshToFoam
./setPatchTypes.sh

simpleFoam > log.run

./postProcess.sh > log.postProcess
