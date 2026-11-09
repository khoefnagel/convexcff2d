"""
Script to compare the pressure and skin friction coefficients over the airfoil
for various boundary treatments at specified grid resolutions. A fine overkill
solution obtained with OpenFOAM is used as reference.

Requires: numpy, matplotlib, glob, json, os
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import os

#================================================================================
#Loading the data from each c* folder

# Cases to be plotted
plotCases = ['CFFVol', 'Chen']

#Initialize dictionaries
allCpDict = {case : [] for case in plotCases}
allCfDict = {case : [] for case in plotCases}
cDict = {case : [] for case in plotCases}

# Read in the data
for case in plotCases:
    for cFolder in np.sort(glob.glob('./c*')):

        #If the Cp file is not there (yet), skip loading the data
        if not os.path.isfile(f'{cFolder}/CFF/{case}_Cp.npy'):
            continue

        #Load surface force coefficients
        Cp = np.load(f'{cFolder}/CFF/{case}_Cp.npy')
        Cf = np.abs(np.load(f'{cFolder}/CFF/{case}_Cf.npy'))

        #Append surface force coefficients to dictionaries
        allCpDict[case].append(Cp)
        allCfDict[case].append(Cf)

        #Load chord length in folder name and append to dictionary
        cDict[case].append(int(cFolder.split('/')[-1][1:]))

#Load airfoil coordinates in the local frame (LE at x=0, TE at =1)
airfoilCoordsLocal = np.load('./c008/CFF/airfoilCoordsLocal.npy')

#================================================================================
#Load OpenFOAM data

#Load the caseDef variables into the current scope
with open('./OpenFOAMFine/caseDef', 'r') as f:
    lines = f.readlines()
for line in lines:
    var, val = line.split(';')[0].split(' ')
    locals().update({var : float(val)})

#Read the header from the Data.csv file
with open('./OpenFOAMFine/Surface_data.csv') as f:
    header = f.readlines()[0][:-2].replace('"', '').split(',')

#Read the data from the Data.csv file
data = np.genfromtxt('./OpenFOAMFine/Surface_data.csv', skip_header=1,
                     delimiter=',')

#Create dictionary of the data, with the header entry as the key
dataDict = {h : d for h, d in zip(header, data.T)}

#Convert pressure and skin friction forces to force coefficients
dataDict['Cp'] = dataDict['p']/(0.5*U**2)
dataDict['Cf'] = np.sqrt(dataDict['wallShearStress:0']**2 +\
                         dataDict['wallShearStress:1']**2 +\
                         dataDict['wallShearStress:2']**2)/(0.5*U**2)

#Create sortInds: sorting the values to follow the airfoil
#This is needed since the raw points are not ordered.
sortInds = [0]
X, Y = dataDict['Points:0'], dataDict['Points:1']
while len(sortInds) < len(X):
    ii = sortInds[-1]
    for ind in np.argsort((X[ii] - X)**2 + (Y[ii] - Y)**2):
        if ind not in sortInds:
            sortInds.append(int(ind))
            break

#================================================================================
#Plotting Cp

plt.rcParams.update({'font.size' : 15}) #Update plot font size
caseInd1 = 1 #Index of the first case to plot
caseInd2 = 3 #Index of the second case to plot
lw = 2

plt.figure()
plt.plot(airfoilCoordsLocal[:,0], allCpDict['CFFVol'][caseInd1], lw=lw,
         label=f'CFF ({cDict[case][caseInd1]} cells per chord)', c='r', zorder=50)
plt.plot(airfoilCoordsLocal[:,0], allCpDict['Chen'][caseInd1], lw=lw,
         label=f'Chen et al. ({cDict[case][caseInd1]} cells per chord)', c='b',
         zorder=100, linestyle=(0,(1,0.5)))
plt.plot(airfoilCoordsLocal[:,0], allCpDict['CFFVol'][caseInd2], lw=lw,
         label=f'CFF ({cDict[case][caseInd2]} cells per chord)', c='g', zorder=50,
         linestyle=(0,(3,1,1,1)))
plt.plot(dataDict['Points:0'][sortInds]/c+0.25, dataDict['Cp'][sortInds], lw=lw,
         c='k', linestyle='dashed', label='OpenFOAM (~500 cells per chord)')
plt.xlabel(r'$x/c$')
plt.ylabel(r'$C_p$')
plt.grid()
plt.xlim(0, 1)
plt.ylim(-1, 1.3)
plt.gca().invert_yaxis()
plt.legend()
plt.tight_layout()
plt.savefig('./Figures/Cp_vs_x.pdf')

#================================================================================
#Plotting Cf

plt.figure()
plt.plot(airfoilCoordsLocal[:,0], allCfDict['CFFVol'][caseInd1], lw=lw,
         label=f'CFF ({cDict[case][caseInd1]} cells per chord)', c='r', zorder=50)
plt.plot(airfoilCoordsLocal[:,0], allCfDict['Chen'][caseInd1], lw=lw,
         label=f'Chen et al. ({cDict[case][caseInd1]} cells per chord)', c='b',
         zorder=100, linestyle=(0,(1,0.5)))
plt.plot(airfoilCoordsLocal[:,0], allCfDict['CFFVol'][caseInd2], lw=lw,
         label=f'CFF ({cDict[case][caseInd2]} cells per chord)', c='g', zorder=50,
         linestyle=(0,(3,1,1,1)))
plt.plot(dataDict['Points:0'][sortInds]/c+0.25, dataDict['Cf'][sortInds], lw=lw,
         c='k', linestyle='dashed', label='OpenFOAM (~500 cells per chord)')
plt.xlabel(r'$x/c$')
plt.ylabel(r'$C_f$')
plt.grid()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('./Figures/Cf_vs_x.pdf')

plt.show()
