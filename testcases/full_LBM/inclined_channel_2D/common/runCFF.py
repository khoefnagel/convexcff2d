"""
Run different boundary treatments for a inclined 2D channel flow driven
by a body force. The following case parameters should be defined in
the file ../caseDef.json:
- N : Number of cell sizes between the two walls
- width : Number of fluid nodes in the x direction
- height : Number of fluid nodes in the y direction
- thetaDeg : Angle of the channel with the x axis in degrees
- g : Body force in the y direction
- tau : Relaxation time
- Nt : Number of timesteps to run

The resulting velocity magnitude fields are saved in files U_<case>.npy
and the time convergence of the maximum velocity is plotted and saved
as ../timeConvergence/<case>_timeConvergence.pdf
Furthermore, the distance to the wall at the cell centroids and cell centers
is computed and saved in wallDistCentroid.npy and wallDistCenter.npy,
respectively. The cell volumes are saved in cellVolumes.npy.

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

# Define geometry
theta = thetaDeg*np.pi/180

b = N//2 # Vertical solid portion on bottom left
a = height - b - N/np.cos(theta) # Vertical solid portion on top left

# Locations where the walls intersect the domain boundaries
x1, x2, x3, x4 = 0, a/np.tan(theta), width-b/np.tan(theta), width
y1, y2, y3, y4 = 0, b, height-a, height

# Define the three polygons that make up the geometry
lowerTriangle = np.array([[x3, y1], [x4, y1], [x4, y2], [x3, y1]])
center = np.array([[x1, y1], [x2, y1], [x4, y3], [x4, y4],
                   [x3, y4], [x1, y2], [x1, y1]],)
upperTriangle = np.array([[x1, y3], [x2, y4], [x1, y4], [x1, y3]])
geom = [lowerTriangle, center, upperTriangle]

# Body force in lattice units
gVec = np.array([-np.cos(theta)*g, -np.sin(theta)*g])

cases = {
    'Chen' : lambda : ConvexChen1998('D2Q9', (height, width), geom,
                                      forceDens=gVec),
    'CFFNone' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 1,
                                      forceDens=gVec, weighing=None),    
    'CFFVol' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 1,
                                     forceDens=gVec, weighing='volume'),
    'CFFOBC' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 1,
                                     forceDens=gVec, weighing='OBCNone'),
    'CFFConst' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 0,
                                       forceDens=gVec),
    'CFFDirNone' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 1,
                                         forceDens=gVec, weighing='CjDirNone'),
    'CFFNoneXY' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 1,
                                        forceDens=gVec, weighing=None, xy=True),
    'CFFVolXY' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 1,
                                     forceDens=gVec, weighing='volume', xy=True),
    'CFFOBCXY' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 1,
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

    # Save velocity field
    np.save(f'{case}_U', S.getMoments()[1])


def getWallDist(X, Y):
    """
    Compute the distance from each point (X, Y) to the nearest wall.
    """

    # Determine if the point is in the lower right or upper left part
    lowerRightInds = np.where(X*np.tan(theta) > Y)
    upperLeftInds = np.where(X*np.tan(theta) <= Y)

    wallDist = np.zeros(X.shape)


    denom = np.sqrt(np.tan(theta)**2 + 1)
    wallDist[lowerRightInds] = (np.tan(theta)*X[lowerRightInds] - Y[lowerRightInds] - a)/denom
    wallDist[upperLeftInds] = (-np.tan(theta)*X[upperLeftInds] + Y[upperLeftInds] - b)/denom
    return wallDist

wallDistCentroid = getWallDist(S.centroidsGlob[0,:,:,0], S.centroidsGlob[0,:,:,1])
wallDistCenter = getWallDist(*np.meshgrid(np.arange(width)+0.5, np.arange(height)+0.5))
np.save('wallDistCentroid', wallDistCentroid)
np.save('wallDistCenter', wallDistCenter)

np.save('cellVolumes', S.cellVolumes)

