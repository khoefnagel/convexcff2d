"""
Script to compare the convergence of the lift and drag coefficients for
various boundary treatments and grid resolutions. A fine overkill solution
obtained with OpenFOAM is used as reference.

Requires: numpy, matplotlib, glob, json, os
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import glob
import json
import os

# Cases to be plotted
plotCases = ['CFFVol', 'Chen']

 #Initialize dictionaries
allClDict = {case : [] for case in plotCases}
allCdDict = {case : [] for case in plotCases}
cDict = {case : [] for case in plotCases}

# Read in the data
for case in plotCases:
    for cFolder in np.sort(glob.glob('./c*/')):

        #If the Cl file is not there (yet), skip loading the data
        if not os.path.isfile(f'{cFolder}/CFF/{case}_Cl.npy'):
            continue

        #Load force coefficients
        Cl = np.load(f'{cFolder}/CFF/{case}_Cl.npy')
        Cd = np.load(f'{cFolder}/CFF/{case}_Cd.npy')

        #Append force coefficients to dictionaries
        allClDict[case].append(float(Cl))
        allCdDict[case].append(float(Cd))

        #Load chord length used and append to dictionary
        with open(f'{cFolder}/caseDef.json') as f:
            cDict[case].append(json.load(f)['c'])

#================================================================================
#Loading OpenFOAM data

#Find the file at the latest time
files = glob.glob(('./OpenFOAMCoarse/postProcessing/'
                   'forceCoeffs1/0/forceCoeffs_*.dat'))
times = [int(file.split('_')[-1].split('.')[0]) for file in files]
latestTimeInd = np.argsort(times)[-1]
latestTimeFile = files[latestTimeInd]

#Load the force coefficients from the latest time file
_, _, CdOF, ClOF, _, _ = np.genfromtxt(latestTimeFile)


#================================================================================
#Lift coefficient plot

plt.rcParams.update({'font.size' : 15}) #Update plot font size

plt.figure()
plt.semilogx(cDict['CFFVol'], allClDict['CFFVol'], label='CFF', c='r', lw=2)
plt.semilogx(cDict['Chen'], allClDict['Chen'], label='Chen et al.', c='b', lw=2,
             linestyle=(0,(1,0.5)), zorder=100)
plt.axhline(ClOF, c='k', label='OpenFOAM (~500 cells per chord)',
            lw=2, linestyle='dashed')
plt.xlabel('Cells per chord')
plt.ylabel(r'$C_l$')
plt.grid()
plt.xticks([8, 16, 32, 64, 128, 256])
plt.gca().set_xticklabels([8, 16, 32, 64, 128, 256])
plt.gca().xaxis.set_minor_locator(ticker.NullLocator())
plt.legend()
plt.tight_layout()
plt.savefig('./Figures/Cl_vs_cpc.pdf')

#================================================================================
#Drag coefficient plot
    
plt.figure()
plt.semilogx(cDict['CFFVol'], allCdDict['CFFVol'], label='CFF', c='r', lw=2)
plt.semilogx(cDict['Chen'], allCdDict['Chen'], label='Chen et al.', c='b', lw=2,
             linestyle=(0,(1,0.5)), zorder=100)
plt.axhline(CdOF, c='k', label='OpenFOAM (~500 cells per chord)',
            lw=2, linestyle='dashed')
plt.xlabel('Cells per chord')
plt.ylabel(r'$C_d$')
plt.grid()
plt.xticks([8, 16, 32, 64, 128, 256])
plt.gca().set_xticklabels([8, 16, 32, 64, 128, 256])
plt.gca().xaxis.set_minor_locator(ticker.NullLocator())
plt.legend()
plt.tight_layout()
plt.savefig('./Figures/Cd_vs_cpc.pdf')


#================================================================================
#Relative convergence of lift coefficient

plt.figure()
ClCFF = np.array(allClDict['CFFVol'])
plt.loglog(cDict['CFFVol'][:-1],
           np.abs(ClCFF[:-1] - ClCFF[1:])/np.abs(ClCFF[1:]),
           label='CFF', c='r', lw=2)
ClChen = np.array(allClDict['Chen'])
plt.loglog(cDict['Chen'][:-1],
           np.abs(ClChen[:-1] - ClChen[1:])/np.abs(ClChen[1:]),
           label='Chen et al.', c='b', lw=2)
plt.axline((8, 0.25), (16, 0.25/4), c='k', lw=2, linestyle='dashed',
           label=r'$x^2$')
plt.axline((8, 0.3), (16, 0.3/2), c='k', lw=2, linestyle='dotted',
           label=r'$x$')
plt.xlabel('Cells per chord (cpc)')
plt.ylabel(r'$|C_l^{(cpc)} - C_l^{(cpc+1)}|/|C_l^{(cpc+1)}|$')
plt.grid()
plt.xticks([8, 16, 32, 64, 128])
plt.gca().set_xticklabels([8, 16, 32, 64, 128])
plt.gca().xaxis.set_minor_locator(ticker.NullLocator())
plt.legend()
plt.tight_layout()
plt.savefig('./Figures/Cl_relconvergence.pdf')

#================================================================================
#Relative convergence of drag coefficient

plt.figure()
CdCFF = np.array(allCdDict['CFFVol'])
plt.loglog(cDict['CFFVol'][:-1],
           np.abs(CdCFF[:-1] - CdCFF[1:])/np.abs(CdCFF[1:]),
           label='CFF', c='r', lw=2)
CdChen = np.array(allCdDict['Chen'])
plt.loglog(cDict['Chen'][:-1],
           np.abs(CdChen[:-1] - CdChen[1:])/np.abs(CdChen[1:]),
           label='Chen et al.', c='b', lw=2)
plt.axline((8, 0.18), (16, 0.18/4), c='k', lw=2, linestyle='dashed',
           label=r'$x^2$')
plt.axline((8, 0.25), (16, 0.25/2), c='k', lw=2, linestyle='dotted',
           label=r'$x$')
plt.xlabel('Cells per chord (cpc)')
plt.ylabel(r'$|C_d^{(cpc)} - C_d^{(cpc+1)}|/|C_d^{(cpc+1)}|$')
plt.grid()
plt.xticks([8, 16, 32, 64, 128])
plt.gca().set_xticklabels([8, 16, 32, 64, 128])
plt.gca().xaxis.set_minor_locator(ticker.NullLocator())
plt.legend()
plt.tight_layout()
plt.savefig('./Figures/Cd_relconvergence.pdf')

plt.show()
