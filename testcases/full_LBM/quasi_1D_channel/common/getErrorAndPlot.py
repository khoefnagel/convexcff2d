"""
Script to compute errors and plot velocity profiles for quasi-1D channel flow case.
The U_<case>.npy files must be present in the CFF folder.
The L1, L2, and LInf errors are saved in errors.json and the plot in Profiles.pdf.

Requires numpy, matplotlib, glob, json
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import json

plt.rcParams.update({'font.size' : 14})

#Load variables defined in the caseDef json file into local namespace
with open('./caseDef.json') as f:
    locals().update(json.load(f))

# Indices to compute errors on: the cut cells are excluded as the physical location
# of the solution of Chen's algorithm is not well defined there.
errorInds = np.index_exp[1:-1]
           

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
xCentroids = np.array([1-pfluid/2, *np.arange(1.5, Nx-1), Nx-1+pfluid/2])
xCellCenters = np.arange(0.5, Nx)
X = {'Chen' : xCellCenters, 'CFFConst' : xCentroids,
     'CFFNone' : xCentroids, 'CFFNoneXY' : xCentroids, 'CFFVol' : xCentroids, 'CFFVolXY' : xCentroids,
     'CFFOBC' : xCentroids, 'CFFOBCXY' : xCentroids}

# Cell volumes, which are used as weights in the L1 and L2 norms
V = np.array([pfluid, *[1]*(Nx-2), pfluid])

#Channel height in lattice units
H = Nx - 2*(1-pfluid)

# Maximum velocity in the channel
UMax = g*H**2/(8*nu)

#Exact velocity, evaluated at the locations in space
Ue = {key : UExact(x - (1-pfluid), H)/UMax for key, x in X.items()}

# Exact solution evaluated at a finer grid for plotting
X['Exact'] = np.linspace(1-pfluid, 1-pfluid+H, 100)
UDict = {'Exact' : UExact(X['Exact'] - (1-pfluid), H)/UMax}

# Load velocity, compute errors and store in errDict
errDict = {}
cases = []
for caseFile in glob.glob('./CFF/U_*.npy'):

    # Get case name
    case = caseFile.split('/')[-1][2:-4]
    cases.append(case)

    UDict[case] = np.load(caseFile)[0,0,5:-5,0]/UMax

    L1Alg = L1(UDict[case][errorInds], Ue[case][errorInds], V[errorInds])
    L2Alg = L2(UDict[case][errorInds], Ue[case][errorInds], V[errorInds])
    LInfAlg = LInf(UDict[case][errorInds], Ue[case][errorInds])
    errDict[f'{case} L1'] = L1Alg
    errDict[f'{case} L2'] = L2Alg
    errDict[f'{case} LInf'] = LInfAlg

with open('./errors.json', 'w') as f:
    json.dump(errDict, f)

# Plot velocity profiles
markers = ['s', 'o', 'D', 'd', '*', '^', 'v']
colors = ['b', 'g', 'y', 'r', 'm', 'orange', 'k']
for j, case in enumerate(cases):

    # Normalized x coordinates for plotting: from -1 to 1 over the channel height
    xNorm = 2/H*(X[case] - (1 - pfluid)) - 1

    plt.plot(xNorm, UDict[case], label=case, marker=markers[j], c=colors[j])

plt.grid()
plt.xlabel(r'$x$/($H$/2)')
plt.ylabel(r'$U_y$/$U_{y,max}$')
plt.legend()
plt.savefig('./Profiles.pdf')
