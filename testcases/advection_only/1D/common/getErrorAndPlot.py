"""
Script to compute errors and plot the populations profile for the advection only
1D case. The F_<case>.npy files must be present in the CFF folder,
as well as the cellCenters.npy and cellVolumes.npy files.
The L1, L2, and LInf errors are saved in populationErrors.json and the plot in Profiles.pdf.

Requires numpy, matplotlib, glob, json, glob
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import glob

plt.rcParams.update({'font.size' : 14})

#Load variables defined in the caseDef json file into local namespace
with open('./caseDef.json') as f:
    locals().update(json.load(f))

# Indices to compute errors on: the cut cells are excluded as the physical location
# of the solution of Chen's algorithm is not well defined there.
errorInds = np.index_exp[1:-1]


# Load the cell volumes
VRaw = np.load('./CFF/cellVolumes.npy')[0,0,:,0]
fluidInds = np.where(VRaw > 1e-10)[0]
V = VRaw[fluidInds]

# Load the cell centers
cellCenters = np.load('./CFF/cellCenters.npy')[0,0,fluidInds,0]

# Exact populations solution
Fe = np.load('./CFF/F_exact.npy')[:,0,fluidInds,0]

# Error norms
L1 = lambda approx, exact, V : np.average(np.abs(approx - exact), weights=V)
L2 = lambda approx, exact, V : np.sqrt(np.sum(V * (approx - exact)**2)/np.sum(V))
LInf = lambda approx, exact : np.max(np.abs(approx - exact))

# Load velocity, compute errors and store in errDict
ErrDict = {}
cases = []
for caseFile in glob.glob('./CFF/F_*.npy'):

    # Get case name
    case = caseFile.split('/')[-1][2:-4]
    cases.append(case)    

    # Load populations
    FAlg = np.load(f'./CFF/F_{case}.npy')[:,0,fluidInds,0]

    ErrDict[f'{case} L1'] = {}
    ErrDict[f'{case} L2'] = {}
    ErrDict[f'{case} LInf'] = {}

    # Compute errors for each population
    for i in range(9): 
        L1Alg = L1(FAlg[i][errorInds], Fe[i][errorInds], V[errorInds])
        L2Alg = L2(FAlg[i][errorInds], Fe[i][errorInds], V[errorInds])
        LInfAlg = LInf(FAlg[i][errorInds], Fe[i][errorInds])
        ErrDict[f'{case} L1'][i] = L1Alg
        ErrDict[f'{case} L2'][i] = L2Alg
        ErrDict[f'{case} LInf'][i] = LInfAlg

    # Plot profile of the f1 population
    plt.plot(cellCenters, FAlg[1], label=case, zorder=100)
plt.plot(cellCenters, Fe[1], label='Exact', c='k')
plt.grid()
plt.legend()
plt.xlabel(r'$x$')
plt.ylabel(r'$f_1$')
plt.tight_layout()
plt.savefig('./f1_profile.pdf')
    
    
with open('./populationErrors.json', 'w') as f:
    json.dump(ErrDict, f)


