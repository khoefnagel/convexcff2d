"""
Script to plot the error norms for all cases in the 2D inclined channel
pure advection testcase. Also computes and prints the observed order of
convergence (OOC) for all cases. The figures are saved in the Figures/ folder.

The orders of convergence are written to orderOfConvergence.txt

Requires matplotlib, numpy, glob, json, os, sys
"""


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import glob
import json
import os
import sys
import numpy as np


plt.rcParams.update({'font.size' : 14})


# Initialize dictionaries to hold all errors
errTypes = ['L1', 'L2', 'LInf']
allPopulationsErrDict = {err : {i : {} for i in range(9)} for err in errTypes}

# Define which population pairs are opposite moving
PopulationsPairDict = {0 : '0',
                   1 : '1-3',
                   2 : '2-4',
                   3 : '1-3',
                   4 : '2-4',
                   5 : '5-7',
                   6 : '6-8',
                   7 : '5-7',
                   8 : '6-8'}
allPopulationsPairErrDict = {err : {iPair : {} for iPair in\
                                set(PopulationsPairDict.values())} for err in errTypes}

# Colors for each of the four opposite moving population pairs
colorDict = {'1-3' : '#009E73',
             '2-4' : '#E69F00',
             '5-7' : '#CC79A7',
             '6-8' : '#0072B2'}

# Read in all errors into allPopulationsErrDict. Structure is:
# allPopulationsErrDict = {'errType' : {iPair : {'case' : [[Delta], [Error]]}}}
for NFolder in glob.glob(f'./N*'):

    # Skip if no errors.json (yet) exists
    if not os.path.isfile(f'{NFolder}/populationErrors.json'):
        continue
    
    # Read in case parameters
    with open(f'{NFolder}/caseDef.json') as f:
        locals().update(json.load(f))

    H = N #Channel height in lattice units
    delta = 1/H #Cell spacing for channel height=1
    
    # Read in errors
    with open(f'{NFolder}/populationErrors.json') as f:
        PopulationsErrDict = json.load(f)

    # Add errors to allPopulationsErrDict
    for case, casePopulationsErrDict in PopulationsErrDict.items():
        for i, PopulationErrDict in casePopulationsErrDict.items():            
            for errType, err in PopulationErrDict.items():
                if case not in allPopulationsErrDict[errType][int(i)].keys():
                    allPopulationsErrDict[errType][int(i)][case] = [[], []]
                allPopulationsErrDict[errType][int(i)][case][0].append(delta)
                allPopulationsErrDict[errType][int(i)][case][1].append(err)

# Compute the average of the error for each opposite moving population pair.
# Store in allPopulationsPairErrDict.
for errType in allPopulationsErrDict.keys():                
    for iPair in set(PopulationsPairDict.values()):
        iPairLst = [int(iPair[0]), int(iPair[-1])]
        i, ii = iPairLst
        for case in allPopulationsErrDict[errType][i].keys():
            delta = allPopulationsErrDict[errType][i][case][0]
            iErr = np.array(allPopulationsErrDict[errType][i][case][1])
            iiErr = np.array(allPopulationsErrDict[errType][ii][case][1])

            if errType == 'L1':
                allPopulationsPairErrDict[errType][iPair][case] = [delta, (iErr + iiErr)/2]
            elif errType == 'L2':
                allPopulationsPairErrDict[errType][iPair][case] = [delta,
                                                            np.sqrt((iErr**2 + iiErr**2)/2)]
            elif errType == 'LInf':
                allPopulationsPairErrDict[errType][iPair][case] = [delta,
                                                            np.max([iErr, iiErr], axis=0)]

# Remove still population
movingPopulationsPairs = set(PopulationsPairDict.values())
movingPopulationsPairs.remove('0')

# Compute and print order of convergence
with open('./orderOfConvergence.txt', 'w') as f:

    # Pipe printed output to file
    sys.stdout = f
    for errType in allPopulationsPairErrDict.keys():
        print(f'Observed Order of Convergence of the {errType}-norm for all cases:')    
        for case in allPopulationsPairErrDict['L1']['1-3'].keys():
            for iPair in movingPopulationsPairs:
                delta = allPopulationsPairErrDict[errType][iPair][case][0]
                Err = allPopulationsPairErrDict[errType][iPair][case][1]
                OOC = np.log(Err[-2]/Err[-1])/np.log(delta[-2]/delta[-1])
                print(f'{case}, population pair {iPair}: {OOC:.02f}')
        print()
sys.stdout = sys.__stdout__
print('Orders of convergence written to orderOfConvergence.txt')


# Plot all cases
plt.rcParams.update({'figure.max_open_warning' : 50})
markers = ['s', 'o', 'D', 'd', '*', '^', '1', 'v']
errLabels = {'L1' : r'$L_1$', 'L2' : r'$L_2$', 'LInf' : r'$L_\infty$'}
linestyles = ['solid', 'dashed']

# Create custom legend handles
customLines = []
customLabels = []

plt.figure()
for j, iPair in enumerate(colorDict.keys()):
    for k, case in enumerate(allPopulationsPairErrDict['L1'][iPair].keys()):
        Delta, E = allPopulationsPairErrDict['L1'][iPair][case]
        sortInds = np.argsort(Delta)[::-1]
        plt.loglog(np.array(Delta)[sortInds], np.array(E)[sortInds],
                   marker=markers[j], c=colorDict[iPair], linestyle=linestyles[k])

        if j == 0:
            customLines.append(Line2D([0], [0], c=colorDict[iPair], linestyle=linestyles[k]))
            customLabels.append({'CFFVol' : 'CFF', 'Chen' : 'Chen et al.'}[case])

    customLines.append(Line2D([0], [0], c=colorDict[iPair], marker=markers[j]))
    customLabels.append(f'i={iPair[0]},{iPair[-1]}')
plt.legend(customLines, customLabels)
plt.grid(which='both')
plt.xticks([0.01, 0.1])
plt.ylabel(r'$L_1$ population error')
plt.xlabel(r'Cell size $\Delta$')
plt.tight_layout()
plt.savefig('./Figures/CyclingSineConvergence.pdf')
plt.show()


