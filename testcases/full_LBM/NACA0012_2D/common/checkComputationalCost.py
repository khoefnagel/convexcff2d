'''
Script to obtain statistics regarding the computational cost per timestep of
the CFF boundary treatment as well as Chen et al. Since this cost depends
directly on the number of nonzeros in a row, the maximum and average number of
nonzeros per row are printed.

Requires: numpy, json, matplotlib, sys
'''


import numpy as np
import json
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '../../../../../../../src/')
from ConvexChen1998 import ConvexChen1998
from ConvexCFF2D import ConvexCFF2D

# Cases to run
runCases = ['CFFVol', 'Chen']


# Load variables defined in the caseDef json file into local namespace
with open('../caseDef.json') as f:
    locals().update(json.load(f))


 #Compute the actual chord Reynolds number and print it for verification
Re_c = (3*U*c)/(tau - 1/2)

#================================================================================
# Reading and processing airfoil coordinates

# Airfoil coordinate file
airfoilFile = './NACA0012.dat'

# Read in and process airfoil coords to form closed shape
# MODIFY IF NECESSARY!!
airfoilRaw = np.genfromtxt(airfoilFile, skip_header=1)
airfoilCoords = np.vstack((airfoilRaw, airfoilRaw[0]))

# Scale, translate, rotate airfoil coords
alpha = np.deg2rad(alphaDeg)
rotationMatrix = np.array([[np.cos(alpha), np.sin(alpha)],
                           [-np.sin(alpha), np.cos(alpha)]])
airfoilCoordsScaled = (airfoilCoords + np.array([-0.25, 0]))*c
airfoilCoordsRotated = np.einsum('ik,jk->ij', airfoilCoordsScaled, rotationMatrix)
airfoilCoordsTrans = airfoilCoordsRotated + np.array([width/4, height/2])

# List of solid objects in the flow (in this case only the airfoil)    
objects = [airfoilCoordsTrans]

# Compute normal and tangent vectors of the airfoil panels
dxdyAirfoil = np.diff(airfoilCoordsTrans, axis=0)
lAirfoil = np.linalg.norm(dxdyAirfoil, axis=1, keepdims=True)
tangentsAirfoil = dxdyAirfoil/lAirfoil
normalsAirfoil = dxdyAirfoil[:,::-1]*np.array([-1, 1])/lAirfoil

# Local (untransformed coordinates)
airfoilCoordsLocal = (airfoilCoords[:-1] + airfoilCoords[1:])/2

# Save untransformed and transformed airfoil coordinates separately
np.save('airfoilCoordsLocal', airfoilCoordsLocal)
np.save('airfoilCoordsGlobal', airfoilCoordsTrans)


#================================================================================
# Predefine cases

# Cases are defined as functions such that instances are only created later
# when the function is case called.
cases = {
    'Chen' : lambda : ConvexChen1998('D2Q9', (height, width), objects),
    'CFFConst' : lambda : ConvexCFF2D('D2Q9', (height, width), objects, 0),
    'CFFNone' : lambda : ConvexCFF2D('D2Q9', (height, width), objects, 1,
                                      weighing=None),    
    'CFFVol' : lambda : ConvexCFF2D('D2Q9', (height, width), objects, 1,
                                     weighing='volume'),
    'CFFOBC' : lambda : ConvexCFF2D('D2Q9', (height, width), objects, 1,
                                     weighing='OBCNone'),
}

#================================================================================
# Main loop


for case in runCases:

    S = cases[case]()

    #Eliminate small values and zeros
    S.SM.data[np.abs(S.SM.data) < 1e-12] = 0
    S.SM.eliminate_zeros()

    #Get number of nonzero entries per row
    rownnz = S.SM.getnnz(axis=1)

    # Rows that correspond to a pgram overlapped cell (i.e. it is not a direct stream)
    boundaryInds = np.where(rownnz > 1+1e-12)[0]

    # Filter out nonmoving populations
    boundaryInds = boundaryInds[np.where(boundaryInds > np.prod(S.N))]

    with open('computationalCostStatistics.txt', 'w') as f:
        # Redirect printing to file
        sys.stdout = f
        
        print()
        print(f'Computational cost statistics for {case}:')
        print(f'Maximum number of nonzeros in a row: {np.max(rownnz[boundaryInds])}')
        print(f'Average number of nonzeros in a row: {np.average(rownnz[boundaryInds])}')
sys.stdout = sys.__stdout__
print('Computational cost statistics written to computationalCostStatistics.txt')

    





