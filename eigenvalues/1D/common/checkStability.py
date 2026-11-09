'''
Script to assess the stability of various versions of the CFF boundary treatment. This
is done by printing the largest eigenvalue magnitude found for the 1D channel
geometry. If the magnitude is larger than 1 (by an amount larger than machine epsilon), 
this means that the boundary treatment is unstable.

The output is written to maxEigMagnitude.txt

Requires: numpy, glob, json, sys
'''

import numpy as np
import glob
import json
import sys


cases = ['CFFVol', 'CFFNone', 'CFFOBC', 'CFFVolXY']

for case in cases:
    magnitudes = []
    for pfluidFolder in glob.glob('./pfluid*'):
        for NFolder in glob.glob(f'{pfluidFolder}/N*'):
            for j, iPairFolder in enumerate(glob.glob(f'{NFolder}/*-*/')):
                iPairStr = iPairFolder.split('/')[-2]

                #Load variables defined in the caseDef json file into local namespace
                with open(f'{iPairFolder}/caseDef.json') as f:
                    locals().update(json.load(f))

                # Load the eigenvalues
                L = np.load(f'{iPairFolder}/RawEigs/{case}_eigenvalues.npy')

                # Compute the magnitude of the eigenvalues
                r = np.sqrt(np.real(L)**2 + np.imag(L)**2)

                magnitudes.append(r)
    with open('maxEigMagnitude.txt', 'a') as f:
        sys.stdout = f
        print(f'{case} maximum eigenvalue magnitude: {np.max(np.concatenate(magnitudes))}')
    sys.stdout = sys.__stdout__

print('Maximum eigenvalue magnitudes written to maxEigMagnitude.txt')
        
