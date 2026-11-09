"""
Module containing the SubZone class, which defines a region within a pgram.
Specifically, a subzone is defined such that it lies in only one cell and
streams exclusively to only one cell.

Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

from Geometry import polygonMoment

class SubZone:
    """Defines a subzone within a pgram.
    
    Attributes:
    ----------
    polyPoints : array_like
        Coordinates of the subzone, shape (5, 2) (closed polygon).
    originCellId : tuple
        Index of the cell in which the SubZone lies before streaming.
    destCellId : tuple
        Index of the cell where the SubZone ends up after streaming.
    polyPointsShifted : array_like
        Subzone coordinates shifted to local coordinates of the origin cell.

    Methods:
    -------
    getMomentCC(degx, degy):
        Calculate the moment relative to the center of the origin cell.
        Specifically, the (degx, degy)-th moment is calculated.
    """

    def __init__(self, polyPoints, originCellId, destCellId):
        self.polyPoints = polyPoints
        self.originCellId = originCellId
        self.destCellId = destCellId

        #Subzone coordinates shifted such that the origin lies at the center
        #of the origin cell. This shift is performed throughout for moment
        #calculations.
        self.polyPointsShifted = self.polyPoints - self.originCellId[::-1] -\
                                 np.array([0.5, 0.5])

    def getMomentCC(self, degx, degy):
        """
        Calculate the moment relative to the center of the origin cell.
        Specifically, the (degx, degy)-th moment is calculated.
        """
        return polygonMoment(self.polyPointsShifted, degx, degy)

    def plot(self, hatch=None, c='k'):
        plt.fill(*self.polyPoints.T, hatch=hatch, c=c, fill=False)


    
    
