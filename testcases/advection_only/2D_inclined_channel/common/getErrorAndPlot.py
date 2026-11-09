"""
Script to compute errors and plot the populations profile for the advection only
2D inclined channel case. The <case>_F.npy files must be present in the CFF folder,
as well as the F_exact.npy and cellVolumes.npy files.
The L1, L2, and LInf errors are saved in populationErrors.json and the plot
of the profile of each population under Figures/<case>_i<i>.pdf.

Requires numpy, matplotlib, glob, json, glob
"""


import numpy as np
import matplotlib.pyplot as plt
import glob
import json

plt.rcParams.update({'font.size' : 14}) #Update font size


#Load variables defined in the caseDef json file into local namespace
with open('./caseDef.json') as f:
    locals().update(json.load(f))

# Load the cell volumes
V = np.load('./CFF/cellVolumes.npy')[0,:,:,0]

errInds = np.where(V > 1 - 1e-10) #Only count fully fluid cells

# Error norms
L1 = lambda approx, exact, V : np.average(np.abs(approx - exact), weights=V)
L2 = lambda approx, exact, V : np.sqrt(np.sum(V * (approx - exact)**2)/np.sum(V))
LInf = lambda approx, exact : np.max(np.abs(approx - exact))

Fe = np.load('./CFF/FExact.npy')

populationErrDict = {}
cases = []
for caseFile in glob.glob('./CFF/*_F.npy'):

    # Get case name
    case = caseFile.split('/')[-1][:-6]
    cases.append(case)   

    populationErrDict[case] = {}
    
    # Load populations
    FAlg = np.load(f'./CFF/{case}_F.npy')

    # Compute errors for each population
    for i in range(9):

        populationErrDict[case][i] = {}
        
        # Compute errors
        L1F = L1(FAlg[i,:,:,0][errInds], Fe[i,:,:,0][errInds],
                    V[errInds])
        L2F = L2(FAlg[i,:,:,0][errInds], Fe[i,:,:,0][errInds],
                    V[errInds])
        LInfF = LInf(FAlg[i,:,:,0][errInds], Fe[i,:,:,0][errInds])

        populationErrDict[case][i]['L1'] = L1F
        populationErrDict[case][i]['L2'] = L2F
        populationErrDict[case][i]['LInf'] = LInfF

        # 2D plot of the populations
        fig, axs = plt.subplots(2)
        # Populations of the case
        FAlgFluid = np.ones((height, width))*np.nan
        FAlgFluid[errInds] = FAlg[i, *errInds, 0]
        axs[0].imshow(FAlgFluid, origin='lower', extent=[0,width,0,height])
        axs[0].set_title(f'{case} f{i}')
        
        # Exact populations
        FExactFluid = np.ones((height, width))*np.nan
        FExactFluid[errInds] = Fe[i, *errInds, 0]
        axs[1].imshow(FExactFluid, origin='lower', extent=[0,width,0,height])
        axs[1].set_title(f'Exact f{i}')
        plt.tight_layout()
        plt.savefig(f'./Figures/{case}_i{i}.pdf')
        plt.clf()

    
with open('./populationErrors.json', 'w') as f:
    json.dump(populationErrDict, f, indent=4)








