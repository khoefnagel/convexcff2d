"""
Module containing the Pgram class, which defines a parallelogram (pgram)
belonging to a certain cut facet.

Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

from Geometry import globToLoc, locToGlob
from Boundary import CutFacet
from SubZone import SubZone


class Pgram(CutFacet):
    """
    Represents a pgram belonging to a certain cut facet.

    Attributes:
    ----------
    cutFacet : CutFacet,
        CutFacet object to which the pgram belongs, see the CutFacet class for more
        information.
    Cj : array_like,
        Velocity vector corresponding to the jth population, the pgram is pointed
        antiparallel to Cj.
    diag : bool,
        Whether the stream is diagonal (i.e. |Cj| > 1).
    nonMoving : bool,
        Whether the stream is nonmoving (i.e. |Cj| = 0).
    degenerate : bool,
        Whether the pgram is degenerate (has zero volume), this is the case when
        the facet is parallel to Cj.
    crossedCells : list,
        List of the cell ids of the cells crossed by the pgram from the facet in
        direction -Cj. Each cell id is an array of shape (2,) with the first entry
        being the y-index and the second entry being the x-index.
    subZones : list of list of SubZone,
        Each pgram consists of various sub-zones, these are defined such that
        each sub-zone lies in only one cell and streams to only one cell. This
        attribute is a list of lists of SubZone objects, where each inner list
        contains the SubZone objects lying in the same cell before streaming.
    """

    def __init__(self, cutFacet, Cj):
        
        self.cutFacet = cutFacet

        CutFacet.__init__(self, cutFacet.start, cutFacet.end)

        self.Cj = Cj
        
        # Whether the stream is diagonal
        self.diag = np.linalg.norm(Cj) > 1 + 1e-10

        # Whether the stream is nonmoving (still populations)
        self.nonMoving = abs(np.linalg.norm(Cj)) < 1e-10

        # Whether the pgram is degenerate (has zero volume)
        self.degenerate = abs(np.dot(self.normal, Cj)) < 1e-14

        self.crossedCells = self._getCrossedCells()
        self.subZones = self._getSubZones()

    def _getCrossedCells(self):
        """
        Get the cell indices of the cells crossed by the pgram.

        Returns
        -------
        crossedCells : list
            List of the cell ids of the cells crossed by the pgram from the facet in
            direction -Cj. Each cell id is an array of shape (2,) with the first entry
            being the y-index and the second entry being the x-index.
        """

        Cx, Cy = self.Cj.squeeze()

        # Add the cell ID of the cell in which the boundary is located.
        # Ordered [yInd, xInd].
        crossedCells = [np.floor([self.mid[1], self.mid[0]]).astype(int)]

        # Duct tape fix for facets exactly on grid lines
        if self.onGridLine:
            crossedCells = [np.floor([self.mid[1]-1e-10*Cy,
                                      self.mid[0]-1e-10*Cx]).astype(int)]

        # Add the diagonal cell for diagonal streams
        if self.diag:

            # Function to calculate the distance to the next integer.
            nextIntDist = lambda x : np.ceil(x) - x

            # Calculate x and y distance to first cell border in -Cj direction.
            xDist, yDist = nextIntDist(-Cx*self.mid[0]), nextIntDist(-Cy*self.mid[1])

            # Index delta wrt facet cell
            delta = np.array([-Cy*(xDist > yDist), -Cx*(xDist < yDist)]).astype(int)

            crossedCells.append(crossedCells[0] + delta)

        # Add the cell where the pgram ends
        if not self.nonMoving:
            crossedCells.append(crossedCells[0] - self.Cj[::-1].astype(int))

        return crossedCells

    def _getSubZones(self):
        """
        Each pgram consists of various subzones. The division is made such that
        each pgram lies in only one cell and streams exclusively to one cell. This
        function finds these subzones as a list of SubZone objects.

        Most calculations are performed in local coordinates, see the globToLoc
        function in the Geometry file for the definition.

        Returns
        -------
        subZones : list
            List of SubZone objects representing the subzones of the pgram.
        """

        #---------------------------------------------------------------------------
        # Transform to local coordinates
        
        # Start point, end point and mid point of cut facet respectively
        x1, y1 = globToLoc(self.start[0], self.start[1], self.Cj)
        x2, y2 = globToLoc(self.end[0], self.end[1], self.Cj)
        xm, ym = globToLoc(self.mid[0], self.mid[1], self.Cj)

        #---------------------------------------------------------------------------
        # Special degenerate case for nonmoving populations; the pgram then is
        # a collapsed polygon of the facet.
        if self.nonMoving:
            polyGlob = np.array([[x1, y1], [x2, y2], [x2, y2], [x1, y1], [x1, y1]])
            self.subZones = [[SubZone(polyGlob, self.crossedCells[0],
                                      self.crossedCells[0])]]
            return

        #---------------------------------------------------------------------------
        # Determination of corner vertices of the sub-zones.

        # y-coordinates of the end of the pgram; x-coordinates still x1 and x2
        ye1, ye2 = y1 + 1, y2 + 1

        # Function to find the y-coordinate after streaming from yIn and reflecting
        # on the boundary at yb.
        reflect = lambda yb, yIn : 2*yb + 1 - yIn

        # The edges of the sub-zones are cell borders and their 'reflections' (see
        # reflect function above) through the boundary. In the following,
        # intersection coordinates with the cell borders and their reflections are
        # found.
        if self.diag: # Diagonal stream
            # Vertical and horizontal cell border intersection respectively
            yv1, yv2 = -np.array([x1, x2]) + np.ceil(xm + ym)
            yh1, yh2 = np.array([x1, x2]) + np.ceil(-xm + ym)

            # Intersection of the reflection of the vertical cell border
            yvR1, yvR2 = reflect(y1, yv1), reflect(y2, yv2)

        else: # Straight stream

            if ym%1 == 0:
                ym += 0.1

            # Horizontal cell border intersection
            yh1, yh2 = np.ceil(ym), np.ceil(ym)


        # Horizontal reflection
        yhR1, yhR2 = reflect(y1, yh1), reflect(y2, yh2)

        #---------------------------------------------------------------------------
        # Determination of the order of the corner vertices

        if self.diag: # Diagonal stream

            # Criterion to determine the position of the facet in the cell
            xmCrit = xm%1 if xm%1<0.5 else xm%1-0.5
            ymCrit = ym%1 if xm%1<0.5 else (ym-0.5)%1 

            # Whether the facet is in the bottom right part of
            # the cell (this is assumed in the analytical relations)
            bottomRight = xmCrit<ymCrit and xmCrit+ymCrit < 1

            # If the facet is in the top left instead, the horizontal and
            # vertical lines are swapped.
            if not bottomRight:
                yh1, yv1 = yv1, yh1
                yh2, yv2 = yv2, yh2
                yhR1, yvR1 = yvR1, yhR1
                yhR2, yvR2 = yvR2, yhR2

                # Adjust the criterion to see where the facet is
                # in the equivalent bottom right portion.
                xmCrit = 0.5-xmCrit
                ymCrit = (ymCrit-0.5)%1

            if ymCrit >= 0.5 and ymCrit>=xmCrit+0.5: #Zone D
                # Order: boundary, vertical, horizontal, horizontal reflection,
                # vertical reflection, end of pgram.
                bounds1 = [[y1, yv1], [yv1, yh1], [yh1, yhR1, yvR1, ye1]]
                bounds2 = [[y2, yv2], [yv2, yh2], [yh2, yhR2, yvR2, ye2]]
                
            elif ymCrit >= 0.5 and ymCrit<xmCrit+0.5: #Zone C
                # Order: boundary, vertical, horizontal reflection, horizontal,
                # vertical reflection, end of pgram.
                bounds1 = [[y1, yv1], [yv1, yhR1, yh1], [yh1, yvR1, ye1]]
                bounds2 = [[y2, yv2], [yv2, yhR2, yh2], [yh2, yvR2, ye2]]

            elif ymCrit < 0.5 and xmCrit+ymCrit>=0.5: #Zone B
                #Order: boundary, horizontal reflection, vertical,
                # vertical reflection, horizontal, end of pgram.
                bounds1 = [[y1, yhR1, yv1], [yv1, yvR1, yh1], [yh1, ye1]]
                bounds2 = [[y2, yhR2, yv2], [yv2, yvR2, yh2], [yh2, ye2]]
                
            elif ymCrit < 0.5 and xmCrit+ymCrit<0.5: #Zone A
                # Order: boundary, horizontal reflection, vertical reflection,
                # vertical, horizontal, end of pgram.
                bounds1 = [[y1, yhR1, yvR1, yv1], [yv1, yh1], [yh1, ye1]]
                bounds2 = [[y2, yhR2, yvR2, yv2], [yv2, yh2], [yh2, ye2]]

        else: # Straight stream
            if ym%1 > 0.5: # Facet in upper half
                bounds1 = [[y1, yh1], [yh1, yhR1, ye1]]
                bounds2 = [[y2, yh2], [yh2, yhR2, ye2]]
            else: # Facet in lower half
                bounds1 = [[y1, yhR1, yh1], [yh1, ye1]]
                bounds2 = [[y2, yhR2, yh2], [yh2, ye2]]

        #---------------------------------------------------------------------------
        # Construction of SubZone objects based on the ordered bounds

        # Cell indices of each subzone before and after streaming respectively
        originInds = sum([[self.crossedCells[i]]*(len(bounds1[i])-1)\
                          for i in range(len(bounds1))], [])
        destInds = originInds[::-1]

        subZones = []
        szi = 0 # Overal subzone index
        for ci in range(len(bounds1)): # Cell index of crossed cell
            subZones.append([])
            for pi in range(len(bounds1[ci])-1): # Subzone index in current cell ci

                # x and y-coordinates of the subzone respectively
                xPolyLoc = [x1, x2, x2, x1, x1]
                yPolyLoc = [bounds1[ci][pi], bounds2[ci][pi], bounds2[ci][pi+1],
                            bounds1[ci][pi+1], bounds1[ci][pi]]

                # Closed polygon of the subzone coordinates
                polyGlob = np.array([[*locToGlob(x, y, self.Cj)]\
                                     for x, y in zip(xPolyLoc, yPolyLoc)])

                # Create SubZone object and append to the list of subzones.
                subZone = SubZone(polyGlob, originInds[szi], destInds[szi])
                subZones[ci].append(subZone)
                
                szi += 1
        return subZones

    def plot(self):
        colors = ['r', 'g', 'b']
        for ci in range(len(self.subZones)):
            c = colors[ci]
            for pi in range(len(self.subZones[ci])):
                self.subZones[ci][pi].plot(c=c)
        plt.gca().set_aspect('equal')


    
    
    
