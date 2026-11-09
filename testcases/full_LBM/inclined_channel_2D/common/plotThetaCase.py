import matplotlib.pyplot as plt
import glob
import json
import os
import numpy as np

plt.rcParams.update({'font.size' : 14})

errTypes = ['L1', 'L2', 'LInf']
allVelErrDict = {err : {} for err in errTypes}
allSpeciesErrDict = {err : {i : {} for i in range(9)} for err in errTypes}

for NFolder in glob.glob(f'./N*'):

    if not os.path.isfile(f'{NFolder}/velocityErrors.json'):
        continue

    with open(f'{NFolder}/caseDef.json') as f:
        locals().update(json.load(f))

    H = N #Channel height in lattice units
    delta = 1/H #Cell spacing for channel height=1

    with open(f'{NFolder}/velocityErrors.json') as f:
        velocityErrDict = json.load(f)

    for case, caseVelErrDict in velocityErrDict.items():
        for errType, err in caseVelErrDict.items():
            if case not in allVelErrDict[errType].keys():
                allVelErrDict[errType][case] = [[], []]

            allVelErrDict[errType][case][0].append(delta)
            allVelErrDict[errType][case][1].append(err)

for NFolder in glob.glob(f'./N*'):

    if not os.path.isfile(f'{NFolder}/speciesErrors.json'):
        continue
    
    with open(f'{NFolder}/caseDef.json') as f:
        locals().update(json.load(f))

    H = N #Channel height in lattice units
    delta = 1/H #Cell spacing for channel height=1
    
    with open(f'{NFolder}/speciesErrors.json') as f:
        speciesErrDict = json.load(f)

    for case, caseSpeciesErrDict in speciesErrDict.items():
        for i, specieErrDict in caseSpeciesErrDict.items():
            for errType, err in specieErrDict.items():
                if case not in allSpeciesErrDict[errType][int(i)].keys():
                    allSpeciesErrDict[errType][int(i)][case] = [[], []]

                allSpeciesErrDict[errType][int(i)][case][0].append(delta)
                allSpeciesErrDict[errType][int(i)][case][1].append(err)
            
plt.rcParams.update({'figure.max_open_warning' : 50})
markers = ['s', 'o', 'D', 'd', '*', '^', '1', 'v']
colors = ['b', 'g', 'y', 'r', 'm', 'orange', 'c', 'k']
errLabels = {'L1' : r'$L_1$', 'L2' : r'$L_2$', 'LInf' : r'$L_\infty$'}
for errType, casesDict in allVelErrDict.items():
    plt.figure()
    for j, (case, (Delta, E)) in enumerate(casesDict.items()):
        sortInds = np.argsort(Delta)[::-1]
        plt.loglog(np.array(Delta)[sortInds], np.array(E)[sortInds], label=case,
                   marker=markers[j], c=colors[j])

    plt.xlabel(r'Cell size: $\Delta$')
    plt.ylabel(errLabels[errType] + ' norm')
    plt.legend()
    plt.grid(which='both')
    plt.xlim([0.01, 0.2])
    plt.ylim([0.0001, 0.4])
    plt.xticks([0.01, 0.1], ['0.01', '0.1'])
    plt.yticks([0.0001, 0.001, 0.01, 0.1], ['0.0001', '0.001', '0.01', '0.1'])
    plt.tight_layout()
    plt.savefig(f'./convergencePlots/velocity/{errType}-norm.pdf')

for errType, speciesDict in allSpeciesErrDict.items():
    for i, specieDict in speciesDict.items():
        plt.figure()
        plt.title(f'i = {i}')
        for j, (case, (Delta, E)) in enumerate(specieDict.items()):
            sortInds = np.argsort(Delta)[::-1]
            plt.loglog(np.array(Delta)[sortInds], np.array(E)[sortInds],
                       label=case, marker=markers[j], c=colors[j])
            convRate = np.log(E[-2]/E[-1])/np.log(Delta[-2]/Delta[-1])
            print(f'{case} {errType} convergence rate i={i} : {convRate}')

        plt.xlabel(r'Cell size: $\Delta$')
        plt.ylabel(errLabels[errType] + ' norm')
        plt.legend()
        plt.grid(which='both')
        plt.xlim([0.01, 0.2])
        plt.xticks([0.01, 0.1], ['0.01', '0.1'])
        '''
        if i == 0:
            plt.ylim([0.00000001, 0.00001])            
            plt.yticks([0.00000001, 0.0000001, 0.000001, 0.00001],
                       [r'$10^{-8}$', r'$10^{-7}$', r'$10^{-6}$', r'$10^{-5}$'])

        else:
            plt.ylim([0.000001, 0.001])
            plt.yticks([0.000001, 0.00001, 0.0001, 0.001],
                       [r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$'])'''
        plt.tight_layout()
        plt.savefig(f'./convergencePlots/species/{i}/{errType}-norm.pdf')

#plt.show()

