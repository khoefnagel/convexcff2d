'''
Script to plot the eigenvalues of the 2D inclined channel case at theta=30 degrees
for N=8 and the CFFVol method.
First, the eigenvalues are plotted sorted on frequency for each of the four
opposite moving population pairs (i=1-3, 2-4, 5-7, 6-8).
Second, the magnitude of all eigenvalues is plotted against their frequency.

Requires: numpy, matplotlib, json, glob, os
'''

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import json
from matplotlib.lines import Line2D

plt.rcParams.update({'font.size' : 15})

#Load variables defined in the caseDef json file into local namespace
with open('./1-3/caseDef.json') as f:
    locals().update(json.load(f))
theta = np.deg2rad(thetaDeg)

case = 'CFFVol'

# Base periods for the four opposite moving population pairs
TBaseDict = {'1-3' : 2*N/np.sin(theta),
             '2-4' : 2*N/np.cos(theta),
             '5-7' : 2*N/(np.sqrt(2)*np.sin(np.pi/4 - theta))\
                     if int(thetaDeg) != 45 else height,
             '6-8' : 2*N/(np.sqrt(2)*np.cos(np.pi/4 - theta))
             }

# Major tick locations to use in the sorted eigenvalue plots for each of the four
# opposite moving population pairs
majorTickDict = {'1-3' : [1, 3, 5, 7, 9, 11, 13, 15],
                 '2-4' : [1, 2, 3, 4, 5, 6, 7, 8, 9],
                 '5-7' : [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
                 '6-8' : [1, 2, 3, 4, 5]
                 }
nFreqLabels = 13

# Legend labels to use for the magnitude vs frequency plot
labelDict = {'1-3' : r'$f_{3/1}$: $\leftarrow\rightarrow$',
             '2-4' : r'$f_{2/4}$: $\uparrow\downarrow$',
             '5-7' : r'$f_{5/7}$: $\nearrow\swarrow$',
             '6-8' : r'$f_{6/8}$: $\nwarrow\searrow$'}


# Colors for each of the four opposite moving population pairs
colorDict = {'1-3' : '#009E73',
             '2-4' : '#E69F00',
             '5-7' : '#CC79A7',
             '6-8' : '#0072B2'}


# Special eigenvalues that are marked a-c in the eigenvalue magnitude plot
specialEigIndDict = {'1-3' : [540],
                     '2-4' : [],
                     '5-7' : [550],
                     '6-8' : [485]}
starLabelInd = 0
starLabels = ['a', 'b', 'c']


xticks = [] # Total list of xticks for the eigs magnitude figure

for j, iPairFolder in enumerate(glob.glob(f'./*-*/')):
    
    iPairStr = iPairFolder.split('/')[-2]

    #Load variables defined in the caseDef json file into local namespace
    with open(f'{iPairFolder}/caseDef.json') as f:
        locals().update(json.load(f))
    iPair = np.array(iPairStr.split('-')).astype(int)

    # Load the eigenvalues
    L = np.load(f'{iPairFolder}/RawEigs/{case}_eigenvalues.npy')

    # Compute the magnitude of the eigenvalues
    r = np.sqrt(np.real(L)**2 + np.imag(L)**2)

    # Compute the frequency f of the eigenvalues
    phi = np.arctan2(np.imag(L), np.real(L))
    f = np.abs(phi)/(2*np.pi)

    # Number of frequencies in the plots
    nFreq = int(TBaseDict[iPairStr]/2+1) 


    # Add points to the eigenvalue magnitude figure
    plt.figure(num=1, figsize=(9,4))
    xticks.append(np.arange(nFreq)/TBaseDict[iPairStr])
    plt.scatter(f, r, s=10, c=colorDict[iPairStr],
                zorder=50, label=labelDict[iPairStr])

    # Add any marked "special" eigenvalues (marked with a star and letter).
    for specialEigInd in specialEigIndDict[iPairStr]:
        plt.scatter(f[specialEigInd], r[specialEigInd], zorder=500,
                    marker='*', s=300, c=colorDict[iPairStr])
        plt.plot([f[specialEigInd], f[specialEigInd] + 0.01],
                 [r[specialEigInd], r[specialEigInd] - 0.1],
                 c=colorDict[iPairStr])
        plt.text(f[specialEigInd] + 0.01, r[specialEigInd] - 0.1,
                 starLabels[starLabelInd], c=colorDict[iPairStr],
                 ha='left', va='top')
        starLabelInd += 1


    # Create the sorted eigenvalue figure (separate figure for each opposite moving population pair)
    plt.figure()
    fSortInd = np.argsort(f) #Sort the eigenvalues

    plt.scatter(f[fSortInd], np.arange(len(f)), zorder=50, s=10, c=colorDict[iPairStr])
    plt.xlim(-0.005, 0.505)
    plt.gca().set_ylim(bottom=0)
    plt.xlabel(r'$f_k=\varphi_k/2\pi$ (Eigenvalue frequency)')
    plt.ylabel('Eigenvalue index')
    plt.gca().set_xticks(np.array(majorTickDict[iPairStr])/TBaseDict[iPairStr])
    plt.gca().set_xticks(np.arange(nFreq)/TBaseDict[iPairStr], minor=True)
    plt.gca().set_xticklabels([rf'$f_{{{f}}}$' for f in majorTickDict[iPairStr]])
    plt.grid(which='both')
    plt.title(rf'{["a", "b", "c", "d"][j]}) $i = {{{iPair[0]}}}$ & $i = {{{iPair[1]}}}$')
    plt.tight_layout()
    plt.savefig(f'./Figures/sortedEigs_{iPairStr}.pdf')



# Add xticks to the eigs magnitude figure
plt.figure(num=1)
for x in [0, 0.5]:
    plt.axvline(x,
                color=plt.rcParams['grid.color'],
                lw=plt.rcParams['grid.linewidth'],
                alpha=plt.rcParams['grid.alpha'], zorder=0)
plt.axhline(y=1, c='k')
plt.xlim(-0.005, 0.505)
plt.gca().set_xticks(np.concatenate(xticks), minor=True)
plt.gca().set_ylim(bottom=0)
plt.grid(which='minor', axis='x')
plt.grid(axis='y')
plt.xlabel(r'$f_k=\varphi_k/2\pi$ (Eigenvalue frequency)')
plt.ylabel(r'$r_k$ (Eigenvalue magnitude)')
plt.legend(markerscale=3)
plt.tight_layout()
plt.savefig('./Figures/eigsMagnitude.pdf', transparent=True)
plt.show()
