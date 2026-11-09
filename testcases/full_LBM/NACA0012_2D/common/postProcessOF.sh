#!/bin/bash

simpleFoam -postProcess -func wallShearStress -noZero -latestTime
simpleFoam -postProcess

foamToVTK -latestTime -noZero

pvpython paraviewProcess.py
