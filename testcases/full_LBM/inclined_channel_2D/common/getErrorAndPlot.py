"""
Script to compute errors and plot velocity profiles for the inclined 2D
channel flow case. The U_<case>.npy files must be present in the CFF folder,
as well as the wallDistCentroid.npy, wallDistCenter.npy, and cellVolumes.npy files.
The L1, L2, and LInf errors are saved in errors.json and the plot in Profiles.pdf.

Requires numpy, matplotlib, glob, json, sys
"""



import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import sys

plt.rcParams.update({'font.size' : 14})

#Load variables defined in the caseDef json file into local namespace
with open('./caseDef.json') as f:
    locals().update(json.load(f))

# Load the wall distances and cell volumes
wallDistCentroid = np.load('./CFF/wallDistCentroid.npy') # Wall distance of the cell centroids
wallDistCenter = np.load('./CFF/wallDistCenter.npy') # Wall distance of the cell centers
V = np.load('./CFF/cellVolumes.npy')[0,:,:,0] # Cell volumes

# Indices to compute errors on: the cut cells are excluded as the physical location
# of the solution of Chen's algorithm is not well defined there.
errorInds = np.where(V > 1 - 1e-10) 

fluidInds = np.where(V > 1e-10)

# Exact solution
nu = (tau-0.5)/3 #Kinematic viscosity
def UExact(y, H):
    return -g/(2*nu)*y**2 + g/(2*nu)*H*y

# Error norms
L1 = lambda approx, exact, V : np.average(np.abs(approx - exact), weights=V)
L2 = lambda approx, exact, V : np.sqrt(np.sum(V * (approx - exact)**2)/np.sum(V))
LInf = lambda approx, exact : np.max(np.abs(approx - exact))

# Solutions of CFF are physically located at cell centroids, while Chen's solution seem
# to be at cell centers. We define the x coordinates accordingly. However, this currently
# does not matter for the error computation as we exclude the cut cells.
X = {'Chen' : wallDistCenter,
     'CFFConst' : wallDistCentroid,
     'CFFNone' : wallDistCentroid,
     'CFFNoneXY' : wallDistCentroid,
     'CFFVol' : wallDistCentroid,
     'CFFVolXY' : wallDistCentroid,
     'CFFOBC' : wallDistCentroid,
     'CFFOBCXY' : wallDistCentroid,
     'CFFDirNone' : wallDistCentroid,
     }

# Channel height in lattice units
H = N

# Maximum velocity in the channel
UMax = g*H**2/(8*nu)

#Exact velocity, evaluated at the locations in space
Ue = {case : UExact(x, H) for case, x in X.items()}

# Exact solution evaluated at a finer grid for plotting
xe = np.linspace(0, H, 100)
UDict = {'Exact' : UExact(xe, H)/UMax}


velocityErrDict = {}
cases = []
for caseFile in glob.glob('./CFF/*_U.npy'):

    # Get case name
    case = caseFile.split('/')[-1][:-6]
    cases.append(case)

    velocityErrDict[case] = {}
    
    UxAlg = np.load(f'./CFF/{case}_U.npy')[0,:,:,0]
    UyAlg = np.load(f'./CFF/{case}_U.npy')[0,:,:,1]
    UAlg = np.sqrt(UxAlg**2 + UyAlg**2)
    UDict[case] = UAlg

    # Compute errors and store in errDict
    L1U = L1(UAlg[errorInds]/UMax, Ue[case][errorInds]/UMax, V[errorInds])
    L2U = L2(UAlg[errorInds]/UMax, Ue[case][errorInds]/UMax, V[errorInds])
    LInfU = LInf(UAlg[errorInds]/UMax, Ue[case][errorInds]/UMax)
    velocityErrDict[case]['L1'] = L1U
    velocityErrDict[case]['L2'] = L2U
    velocityErrDict[case]['LInf'] = LInfU

with open('./velocityErrors.json', 'w') as f:
    json.dump(velocityErrDict, f, indent=4)
    
# Plot velocity profiles
markers = ['s', 'o', 'D', 'd', '*', '^', '1', 'v']
colors = ['b', 'g', 'y', 'r', 'm', 'orange', 'c', 'k']
for j, case in enumerate(cases):
    xNorm = X[case]/H - 0.5
    U = UDict[case]/UMax

    plt.scatter(xNorm[fluidInds], U[fluidInds], label=case,
                marker=markers[j], c=colors[j], zorder=10)

plt.plot(xe/H - 0.5, UDict['Exact'], c='k')
plt.grid()
plt.xlabel(r'$x$/($H$/2)')
plt.ylabel(r'$U_y$/$U_{y,max}$')
plt.legend()
plt.savefig('./Profiles.pdf')


