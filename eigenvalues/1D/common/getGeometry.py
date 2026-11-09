'''
Script to get the geometry of the 1D channel.

Requires: numpy
'''

import numpy as np

def getGeometry(pfluid, Nx):
    Ny = 1

    # Note that 5 cells are added to the left and right of the domain
    # such that CFF does not use cells from the other side of the channel.
    walls = [np.array([[-0.5, -0.5], [6-pfluid, -0.5],
                       [6-pfluid, Ny+0.5], [-0.5, Ny+0.5], [-0.5, -0.5]]),
             np.array([[Nx+4+pfluid, -0.5], [Nx+10.5, -0.5],
                       [Nx+10.5, Ny+0.5], [Nx+4+pfluid, Ny+0.5],
                       [Nx+4+pfluid, -0.5]])]
    return walls
