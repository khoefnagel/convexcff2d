"""
Run different boundary treatments for a quasi-1D channel flow driven
by a body force. The following case parameters should be defined in
the file ../caseDef.json:
- Nx : Number of fluid nodes in the x direction
- pfluid : Fluid fraction of the cut cells
- g : Body force in the y direction
- tau : Relaxation time
- Nt : Number of timesteps to run

The resulting velocity magnitude fields are saved in files U_<case>.npy
and the time convergence of the maximum velocity is plotted and saved
as ../timeConvergence/<case>_timeConvergence.pdf

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

# Body force in lattice units
gVec = np.array([0, g])

cases = {
    'Chen' : lambda : ConvexChen1998('D2Q9', (Ny, Nx+10), walls, forceDens=gVec),
    'CFFConst' : lambda : ConvexCFF2D('D2Q9', (Ny, Nx+10), walls, 0,
                                       forceDens=gVec, weighing=None),
    'CFFNone' : lambda : ConvexCFF2D('D2Q9', (Ny, Nx+10), walls, 1,
                                      forceDens=gVec, weighing=None),    
    'CFFVol' : lambda : ConvexCFF2D('D2Q9', (Ny, Nx+10), walls, 1,
                                     forceDens=gVec, weighing='volume'),
    'CFFOBC' : lambda : ConvexCFF2D('D2Q9', (Ny, Nx+10), walls, 1,
                                     forceDens=gVec, weighing='OBCNone'),
    'CFFNoneXY' : lambda : ConvexCFF2D('D2Q9', (Ny, Nx+10), walls, 1,
                                        forceDens=gVec, weighing=None, xy=True), 
    'CFFVolXY' : lambda : ConvexCFF2D('D2Q9', (Ny, Nx+10), walls, 1,
                                     forceDens=gVec, weighing='volume', xy=True),
    'CFFOBCXY' : lambda : ConvexCFF2D('D2Q9', (Ny, Nx+10), walls, 1,
                                     forceDens=gVec, weighing='OBCNone', xy=True),
}


# These are used to monitor time convergence
# We save at most 50 snapshots
nTimeSnapshots = min(50, Nt)
timeConvergenceDict = {case : [] for case in runCases}
saveTimes = np.linspace(0, Nt-1, nTimeSnapshots, dtype=int)


for case in runCases:

    S = cases[case]()
    
    # Initialize to zero velocity and unit density
    rho0 = np.ones((1, *S.N, 1))
    U0 = np.zeros((1, *S.N, 2))
    S.initialize(rho0, U0)

    for it in range(Nt):
        
        S.collide(tau)
        S.stream()

        # Monitor convergence of the maximum velocity in time
        if it in saveTimes:
            UMax = np.max(np.linalg.norm(S.getMoments()[1], axis=-1))
            timeConvergenceDict[case].append(UMax)

    # Plot time convergence
    plt.plot(saveTimes, timeConvergenceDict[case])
    plt.xlabel('Timestep')
    plt.ylabel('Maximum velocity')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'../timeConvergence/{case}_timeConvergence.pdf')
    plt.clf()
    
    # Save velocity magnitude field
    UMag = np.linalg.norm(S.getMoments()[1], axis=-1, keepdims=True)
    np.save(f'U_{case}', UMag)

    




