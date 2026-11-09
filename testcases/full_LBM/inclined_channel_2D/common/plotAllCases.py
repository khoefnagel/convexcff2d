"""
Script to plot the error norms for all cases in the inclined 2D channel testcase.
Also computes and prints the observed order of convergence for all cases.
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
    #[Case, theta, color, marker, label, zorder]
    ['CFFVol', '10', 'r', '+', r'CFF, $theta=10^\circ$', 100],
    ['CFFVol', '20', 'b', '1', r'CFF, $theta=20^\circ$', 100],
    ['CFFVol', '30', 'm', 'd', r'CFF, $theta=30^\circ$', 50],
    ['CFFVol', '40', 'deeppink', 'D', r'CFF, $theta=40^\circ$', 20],
    ['CFFVol', '45', 'c', '^', r'CFF, $theta=45^\circ$', 30],
    #['Chen', '10', 'b', 's', r'Chen, $theta=20^\circ$', 10],
    ['Chen', '20', 'b', 's', r'Chen, $theta=20^\circ$', 10],
    #['Chen', '30', 'b', 's', r'Chen, $theta=20^\circ$', 10],
    #['Chen', '40', 'b', 's', r'Chen, $theta=20^\circ$', 10],
]

# Cases to compute order of convergence for
OOCCases = [
    ['Chen', '10'],
    ['Chen', '20'],
    ['Chen', '30'],
    ['Chen', '40'],
    ['Chen', '45'],
    ['CFFVol', '10'],
    ['CFFVol', '20'],
    ['CFFVol', '30'],
    ['CFFVol', '40'],
    ['CFFVol', '45'],
]  


# Read in all errors into allErrDict. Structure is:
# allErrDict = {'errType' : {'pfluid' : {'case' : [[Delta], [Error]]}}}
allErrDict = {err : {} for err in ['L1', 'L2', 'LInf']}
for thetaFolder in glob.glob(f'./theta*'):
    thetaStr = thetaFolder.split('theta')[-1]
    for err in allErrDict.keys():
        allErrDict[err][thetaStr] = {}
    for NFolder in glob.glob(f'{thetaFolder}/N*'):

        # Skip if no velocityErrors.json (yet) exists
        if not os.path.isfile(f'{NFolder}/velocityErrors.json'):
            continue

        N = float(NFolder.split('N')[-1])

        # Read in case parameters
        with open(f'{NFolder}/caseDef.json') as f:
            locals().update(json.load(f))

        delta = 1/N #Cell spacing for channel height=1

        # Read in errors
        with open(f'{NFolder}/velocityErrors.json') as f:
            caseErrDict = json.load(f)

        # Add errors to allErrDict
        for case, errDict in caseErrDict.items():
            for errType, err in errDict.items():
                if case not in allErrDict[errType][thetaStr].keys():
                    allErrDict[errType][thetaStr][case] = [[], []]

                allErrDict[errType][thetaStr][case][0].append(delta)
                allErrDict[errType][thetaStr][case][1].append(err)

# Compute and print OOC for all cases in OOCCases
with open('orderOfConvergence.txt', 'w') as f:

    # Pipe printed output to file
    sys.stdout = f
    for errType in allErrDict.keys():
        print(f'Observed Order of Convergence of the {errType}-norm for all cases:')    
        for case, thetaStr in OOCCases:
            delta = allErrDict[errType][thetaStr][case][0]
            Err = allErrDict[errType][thetaStr][case][1]
            OOC, _ = np.polyfit(np.log(delta), np.log(Err), 1)
            print(f'{case}, theta {thetaStr}: {OOC:.02f}')
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
    plt.ylabel(errLabels[errType] + ' error norm of velocity magnitude')
    plt.legend(ncol=2, loc=(0.02,0.95), fancybox=True, shadow=True)
    plt.grid(which='both')
    plt.xlim([0.01, 0.2])
    plt.ylim([0.0005, 0.7])
    plt.xticks([0.01, 0.1], ['0.01', '0.1'])
    plt.yticks([0.001, 0.01, 0.1], ['0.001', '0.01', '0.1'])
    plt.tight_layout()
    plt.savefig(f'./Figures/allCasesInclinedChannel{errType}.pdf')
plt.show()
    
