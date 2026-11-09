"""
Script to plot the pressure coefficient distribution over the airfoil surface
as a colorized line.

Requires: numpy, matplotlib, json
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import json

plt.rcParams.update({'font.size' : 14})

# Load variables defined in the caseDef json file into local namespace
with open('../caseDef.json') as f:
    locals().update(json.load(f))


Cp = np.load('./CFFVol_Cp.npy')
AF_coords = np.load('./airfoilCoordsGlobal.npy')

# Create line segments so that we can color them individually
AF_coords2 = np.reshape(AF_coords, (-1, 1, 2))
segments = np.concatenate([AF_coords2[:-1], AF_coords2[1:]], axis=1)

# Create a continuous norm to map from data points to colors
lc = LineCollection(segments, cmap='jet',
                    norm=plt.Normalize(vmin=-1, vmax=-0))
lc.set_array(Cp)
lc.set_linewidth(5)
lc.set_zorder(100)

fig, ax = plt.subplots(figsize=(5.2,2))
ax.add_collection(lc)
ax.set_aspect('equal')

ax.set_xlim(AF_coords[:,0].min() - 0.5,
            (AF_coords[:,0].min() + AF_coords[:,0].max())/2 - 2)
ax.set_ylim(AF_coords[:,1].min() - 0.2, AF_coords[:,1].max() + 0.2)

plt.colorbar(lc, ax=ax, label=r'$C_p$')

# Add grid lines
for xi in range(int(np.floor(min(AF_coords[:,0]))),
                int(np.ceil(max(AF_coords[:,0])))):
    plt.axvline(xi, c='k')

for yi in range(int(np.floor(min(AF_coords[:,1]))),
                int(np.ceil(max(AF_coords[:,1])))):
    plt.axhline(yi, c='k')

plt.axis('off')
plt.tight_layout()
plt.savefig(f'../../Figures/colorizedCp_{c}cpc.pdf')
plt.show()
