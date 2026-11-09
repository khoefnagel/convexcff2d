'''
Script to get the geometry of the 2D inclined channel.

Requires: numpy
'''

import numpy as np


def getGeometry(N, height, width, thetaDeg):
        ''' Get the geometry of the 2D inclined channel.
        
        Parameters
        ----------
        N : int
            Number of cell widths between the channel walls.
        height : float
            Height of the domain.
        width : float
            Width of the domain.
        thetaDeg : float
            Angle of inclination in degrees.
        '''
        theta = thetaDeg*np.pi/180

        b = N//2 # Vertical solid portion on bottom left
        a = height - b - N/np.cos(theta) # Vertical solid portion on top left

        # Locations where the walls intersect the domain boundaries
        x1, x2, x3, x4 = 0, a/np.tan(theta), width-b/np.tan(theta), width
        y1, y2, y3, y4 = 0, b, height-a, height

        # Define the three polygons that make up the geometry
        lowerTriangle = np.array([[x3, y1], [x4, y1], [x4, y2], [x3, y1]])
        center = np.array([[x1, y1], [x2, y1], [x4, y3], [x4, y4],
                           [x3, y4], [x1, y2], [x1, y1]],)
        upperTriangle = np.array([[x1, y3], [x2, y4], [x1, y4], [x1, y3]])

        geom = [lowerTriangle, center, upperTriangle]
        return geom
