import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
from getGeometry import getGeometry

plt.rcParams.update({'font.size' : 18})

X = np.load('./RawEigs/CFFVol_eigenvectors.npy')
fluidInds = np.load('./RawEigs/CFFVol_fluidInds.npy')

#Load variables defined in the caseDef json file into local namespace
with open('./caseDef.json') as f:
    locals().update(json.load(f))

geom = getGeometry(N, height, width, thetaDeg)

eigInd = {'1-3' : 540, '6-8' : 485, '5-7' : 550}[iPairStr]
label = {'1-3' : r'Re$(f_1)$', '6-8' : r'Re$(f_6)$', '5-7' : r'Re$(f_5)$'}[iPairStr]

x2D = np.ones((height, width))*np.nan
x2D[np.unravel_index(fluidInds, (height, width))] =\
                                np.real(X[:len(fluidInds), eigInd])

fig, ax = plt.subplots(figsize=(6,3.5))
im = ax.imshow(x2D, origin='lower', extent=(0,width,0,height), cmap='viridis')

for geo in geom:
    ax.fill(*geo.T, c='k')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax)

cbar.ax.set_title(label, pad=10, loc='left')

ax.axis('off')

plt.tight_layout(pad=0)

plt.savefig(f'../Figures/eigenmode_{iPairStr}.pdf')

plt.show()




