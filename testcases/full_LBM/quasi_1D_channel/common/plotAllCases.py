"""
Script to plot the error norms for all cases in the quasi-1D channel testcase.
Also computes and prints the observed order of convergence (OOC) for all cases.
The figures are saved in the Figures/ folder.

The orders of convergence are written to orderOfConvergence.txt

Requires matplotlib, numpy, glob, json, os, sys
"""

import matplotlib.pyplot as plt
import glob
import json
import numpy as np
import os
import sys

plt.rcParams.update({'font.size' : 14})

# Cases to plot, with plotting details
plotCases = [
    #[Case, pfluid, color, marker, label, zorder]
    ['CFFVol', '1/9', 'r', '+', r'CFF, $p_{fluid}=$1/9', 100],
    ['CFFVol', '1/3', 'g', '1', r'CFF, $p_{fluid}=$1/3', 200],
    ['CFFVol', '1/2', 'm', 'd', r'CFF, $p_{fluid}=$1/2', 50],
    ['CFFVol', '2/3', 'deeppink', 'D', r'CFF, $p_{fluid}=$2/3', 30],
    ['CFFVol', '8/9', 'c', '^', r'CFF, $p_{fluid}=$8/9', 60],
    ['CFFVol', '1/1', 'k', 'o', r'CFF, $p_{fluid}=$1/1', 40],
    ['Chen', '1/3', 'b', 's', r'Chen, $p_{fluid}=$1/3', 20],
]

# Cases to compute order of convergence for
OOCCases = [
    ['Chen', '1/9'],
    ['Chen', '1/3'],
    ['Chen', '1/2'],
    ['Chen', '2/3'],
    ['Chen', '8/9'],
    ['Chen', '1/1'],
    ['CFFVol', '1/9'],
    ['CFFVol', '1/3'],
    ['CFFVol', '1/2'],
    ['CFFVol', '2/3'],
    ['CFFVol', '8/9'],
    ['CFFVol', '1/1'],
]  

# Read in all errors into allErrDict. Structure is:
# allErrDict = {'errType' : {'pfluid' : {'case' : [[Delta], [Error]]}}}
allErrDict = {err : {} for err in ['L1', 'L2', 'LInf']}
for pfluidFolder in glob.glob(f'./pfluid*'):

    pfluidStr = pfluidFolder[-3:].replace('-', '/')
    for err in allErrDict.keys():
        allErrDict[err][pfluidStr] = {}

    for NFolder in glob.glob(f'{pfluidFolder}/N*'):

        # Skip if no errors.json (yet) exists
        if not os.path.isfile(f'{NFolder}/errors.json'):
            continue

        N = float(NFolder.split('N')[-1])

        # Read in case parameters
        with open(f'{NFolder}/caseDef.json') as f:
            locals().update(json.load(f))

        H = Nx - 2 + 2*pfluid #Channel height in lattice units
        delta = 1/H #Cell spacing for channel height=1

        # Read in errors
        with open(f'{NFolder}/errors.json') as f:
            caseErrDict = json.load(f)

        # Add errors to allErrDict
        for caseAndErr in caseErrDict.keys():
            case, errType = caseAndErr.split(' ')
            if case not in allErrDict[errType][pfluidStr].keys():
                allErrDict[errType][pfluidStr][case] = [[], []]

            allErrDict[errType][pfluidStr][case][0].append(delta)
            allErrDict[errType][pfluidStr][case][1].append(caseErrDict[caseAndErr])

# Compute and print OOC for all cases in OOCCases
with open('./orderOfConvergence.txt', 'w') as f:

    # Pipe printed output to file
    sys.stdout = f
    for errType in allErrDict.keys():
        print(f'Observed Order of Convergence of the {errType}-norm for all cases:')
        for case, pfluidStr in OOCCases:
            delta = allErrDict[errType][pfluidStr][case][0]
            L1Err = allErrDict[errType][pfluidStr][case][1]
            OOC, _ = np.polyfit(np.log(delta), np.log(L1Err), 1)
            print(f'{case}, pfluid {pfluidStr}: {OOC:.02f}')
        print()
sys.stdout = sys.__stdout__
print('Orders of convergence written to orderOfConvergence.txt')


# Plot all cases
errLabels = {'L1' : r'$L_1$', 'L2' : r'$L_2$', 'LInf' : r'$L_\infty$'}
for ei, errType in enumerate(allErrDict.keys()):
    plt.figure(num=ei+1)
    for j, (case, pfluidStr, c, marker, label, zorder) in enumerate(plotCases):
        Delta, E = allErrDict[errType][pfluidStr][case]
        sortInds = np.argsort(Delta)[::-1]
        plt.loglog(np.array(Delta)[sortInds], np.array(E)[sortInds], label=label,
                   marker=marker, c=c, zorder=zorder)

    plt.xlabel(r'Cell size: $\Delta$')
    plt.ylabel(errLabels[errType] + ' error norm of y-velocity')
    plt.legend(ncol=2, loc=(0.02,0.95), fancybox=True, shadow=True)
    plt.grid(which='both')
    plt.xlim([0.01, 0.2])
    plt.ylim([0.0001, 0.4])
    plt.xticks([0.01, 0.1], ['0.01', '0.1'])
    plt.yticks([0.0001, 0.001, 0.01, 0.1], ['0.0001', '0.001', '0.01', '0.1'])
    plt.tight_layout()
    plt.savefig(f'./Figures/allCasesGravityChannel{errType}.pdf')
plt.show()
    
