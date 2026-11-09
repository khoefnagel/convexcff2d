'''
Script to generate and save eigenvalues and eigenvectors for the inclined channel
case. 

Requires: numpy, json, sys
'''

import numpy as np
import json
import sys

sys.path.insert(0, '../../../../../../src')
from ConvexChen1998 import ConvexChen1998
from ConvexCFF2D import ConvexCFF2D

# Import function to generate geometry of the inclined channel
from getGeometry import getGeometry

runCases = ['CFFVol', 'CFFNone', 'CFFOBC', 'CFFVolXY']


#Load variables defined in the caseDef json file into local namespace
with open('./caseDef.json') as f:
    locals().update(json.load(f))
theta = np.deg2rad(thetaDeg)
iPair = np.array(iPairStr.split('-')).astype(int)

geom = getGeometry(N, height, width, thetaDeg)


cases = {
    'Chen' : lambda : ConvexChen1998('D2Q9', (height, width), geom),
    'CFFConst' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 0,
                                       weighing=None),
    'CFFNone' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 1,
                                      weighing=None),    
    'CFFVol' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 1,
                                     weighing='volume'),
    'CFFOBC' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 1,
                                     weighing='OBCNone'),
    'CFFDirNone' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 1,
                                        weighing='CjDirNone'),
    'CFFNoneXY' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 1,
                                        weighing=None, xy=True),
    'CFFVolXY' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 1,
                                       weighing='volume', xy=True),
    'CFFOBCXY' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 1,
                                       weighing='OBCNone', xy=True),
}

for case in runCases:
    S = cases[case]()

    # Indices of non-solid cells
    fluidInds = np.where(S.cellVolumes.flatten() > 1e-10)[0]

    # Indices where to slice the streaming matrix: only non-solid cells of the
    # two opposite moving populations in iPair are kept.
    sliceInds = np.hstack((iPair[0]*np.prod(S.N) + fluidInds,
                           iPair[1]*np.prod(S.N) + fluidInds))

    E = np.linalg.eig(S.SM[sliceInds][:,sliceInds].todense())
    L = E.eigenvalues
    X = E.eigenvectors

    np.save(f'./RawEigs/{case}_eigenvalues', L)
    np.save(f'./RawEigs/{case}_eigenvectors', X)
    np.save(f'./RawEigs/{case}_fluidInds', fluidInds)
    
