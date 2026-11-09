"""
Run different boundary treatments for the case of pure advection in a 1D domain. 
The following case parameters should be defined in the file ../caseDef.json:
- Nx : Number of fluid nodes in the x direction
- pfluid : Fluid fraction of the cut cells
- Nt : Number of timesteps to run

The population fields are saved in files F_<case>.npy

Requires: numpy, matplotlib, json, sys
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '../../../../../../../src/')
from ConvexChen1998 import ConvexChen1998
from ConvexCFF2D import ConvexCFF2D

# Cases to run
runCases = ['Chen', 'CFFVol']

#Load variables defined in the caseDef json file into local namespace
with open('../caseDef.json') as f:
    locals().update(json.load(f))


# Define geometry. Note that 5 cells are added at each side in x to
# prevent taking cells on the other side of the domain as neighbors.
Ny = 1
walls = [np.array([[-0.5, -0.5], [6-pfluid, -0.5],
                   [6-pfluid, Ny+0.5], [-0.5, Ny+0.5], [-0.5, -0.5]]),
         np.array([[Nx+4+pfluid, -0.5], [Nx+10.5, -0.5],
                   [Nx+10.5, Ny+0.5], [Nx+4+pfluid, Ny+0.5],
                   [Nx+4+pfluid, -0.5]])]


cases = {
    'Chen' : lambda : ConvexChen1998('D2Q9', (Ny, Nx+10), walls),
    'CFFConst' : lambda : ConvexCFF2D('D2Q9', (Ny, Nx+10), walls, 0, weighing=None),
    'CFFNone' : lambda : ConvexCFF2D('D2Q9', (Ny, Nx+10), walls, 1, weighing=None),    
    'CFFVol' : lambda : ConvexCFF2D('D2Q9', (Ny, Nx+10), walls, 1, weighing='volume'),
    'CFFOBC' : lambda : ConvexCFF2D('D2Q9', (Ny, Nx+10), walls, 1, weighing='OBCNone'),
    'CFFNoneXY' : lambda : ConvexCFF2D('D2Q9', (Ny, Nx+10), walls, 1,
                                        weighing=None, xy=True), 
    'CFFVolXY' : lambda : ConvexCFF2D('D2Q9', (Ny, Nx+10), walls, 1,
                                     weighing='volume', xy=True),
    'CFFOBCXY' : lambda : ConvexCFF2D('D2Q9', (Ny, Nx+10), walls, 1,
                                     weighing='OBCNone', xy=True),
}


# Exact solution function
def getFExact(X, lambdaX, phiX, t, Cj):
    return 0.5 - 0.5*np.cos(2*np.pi/lambdaX*(X + phiX - t*Cj))


#Wavelength and phase of the exact function
lambdaX = (Nx - 2 + 2*pfluid)
phiX = -5 - (1 - pfluid)

for ci, case in enumerate(runCases):

    #----------------------------------------------------------------------------
    #Initialization

    #Initialize the species object
    S = cases[case]()

    #x-coordinates
    X = S.centroidsGlob[0,:,:,0]

    #Initialize F using exact function at t=0
    S.F[:,:,:,0] = getFExact(X, lambdaX, phiX, 0, 0)

    #----------------------------------------------------------------------------
    #Obtaining and saving exact solution
                            
    #Save exact solution, cellcenters and volumes only for the first case (these
    #are the same for all cases).
    if ci == 0:
        FExact = np.zeros(S.F.shape)
        for i in range(S.Q):
            FExact[i,0,:,0] = 0.5-0.5*np.cos(2*np.pi/lambdaX*(X + phiX -\
                                                              Nt*S.Cij[i,0,0,0]))
        np.save('F_exact', FExact)
        np.save('cellCenters', S.centroidsGlob)
        np.save('cellVolumes', S.cellVolumes)
    
    #----------------------------------------------------------------------------
    #Running and saving the case

    for it in range(Nt):

        #Only streaming, no collision
        S.stream()
    
    np.save(f'F_{case}', S.F)


