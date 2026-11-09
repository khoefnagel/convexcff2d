"""
Module containing the Facet and Facets classes, representing a single facet
and a collection of facets, respectively.

Requires: numpy
"""

import numpy as np

from getKnots2D import getKnots2D

class Facet:
    """Class representing a single facet (boundary segment) defined by its start and end
    coordinates. The normal and tangential vectors are computed such that when traveling
    from start to end, the normal vector is on the right:

    start -->--->--⇓-->---> end
                   ⇓normal        
    
    Attributes:
    ----------
    start : array_like,
        Start coordinates (x, y) of facet.
    end : array_like,
        End coordinates (x, y) of facet.
    normal : array_like,
        Normal vector of the facet.
    tangential : array_like,
        Tangential vector of the facet.
    mid : array_like,
        Midpoint of the facet.
    L : float,
        Length of the facet.
    onGridLine : bool,
        Whether the facet lies on a grid line.

    Methods:
    -------
    getCutFacets():
        Cuts the facets on the subgrid lines. Returns a list of CutFacet objects resulting
        from the cut.    
    """

    def __init__(self, start, end):

        self.start = start.flatten()
        self.end = end.flatten()
        
        # Facet length
        self.L = np.linalg.norm(end - start)
        
        if self.L == 0: # Skip the rest of the initialization if facet length is 0
            return

        # Midpoint
        self.mid = (start + end)*0.5

        # Normal vector
        self.normal = (end - start)[::-1]*np.array([1, -1])/self.L

        # Tangential vector
        self.tangential = (end - start)/self.L

        # Whether the facet intersects a grid-line
        self.onGridLine = \
                        (self.start[0] == self.end[0] and self.start[0]%1 == 0) or\
                        (self.start[1] == self.end[1] and self.start[1]%1 == 0)
        return

    def getCutFacets(self):
        """
        Cut the facet on the subgrid lines. Returns a list of CutFacet objects resulting
        from the cut. See getKnots2D for the definition of the subgrid lines.

        Returns
        -------
        cutFacets : list of CutFacet
            List of CutFacet objects resulting from cutting the facet on the subgrid lines.
        """

        # Points where the facet intersects the aforementioned grid lines
        knots = getKnots2D(self.start[0], self.start[1], self.end[0], self.end[1])

        # Add facets between these knots to the list of cut facets
        cutFacets = []
        for i in range(len(knots)-1):
            cutFacet = CutFacet(knots[i], knots[i+1])
            if cutFacet.L < 1e-12: #Skip facets with zero length.
                continue
            cutFacets.append(cutFacet)
        return cutFacets

    def __repr__(self):
        return f"Facet from ({self.start[0]}, {self.start[1]}) to "\
               f"({self.end[0]}, {self.end[1]})."

class CutFacet(Facet):
    """
    Same as Facet, but throws error when trying to obtain cut facets.
    """

    def __init__(self, start, end):
        Facet.__init__(self, start, end)

    def getCutFacets(self):
        raise Exception('Facet already cut.')


class Facets:
    """
    Class representing a collection of facets defined by a list of closed polygons.
    Each polygon defines a solid object.

    Attributes:
    ----------
    solids : list of array_like
        List of closed polygons representing the solid objects. Each polygon is an array
        of shape (Nv, 2) where Nv is the number of vertices.

    allFacets : list of list of Facet
        Nested list containing the Facet objects for each solid object. The outer list
        corresponds to the solids, the inner list to the facets of that solid. These
        facets are uncut.
    allCutFacets : list of list of CutFacet
        Nested list containing the CutFacet objects for each solid object. The outer list
        corresponds to the solids, the inner list to the facets of that solid. These
        facets are cut on the subgrid lines.
    """

    def __init__(self, solids):
        '''Input:
        -solids : list of closed polygons representing the solid objects.'''
        
        #Make sure solids are a list
        if not isinstance(solids, list):
            raise Exception('solids should be a list of closed Polygons.')

        #Check if the solids are closed.
        for solid in solids:
            if np.linalg.norm(solid[0] - solid[-1]) > 1e-12:
                raise Exception('Each solid should be a closed Polygon.')

        self.solids = solids
        
        self.allFacets = self._getAllFacets()
        self.allCutFacets = self._getAllCutFacets()

    def _getAllFacets(self):
        """
        Convert the edges defined in solids to (uncut) Facets, these
        are saved in the nested list allFacets.
        """
        allFacets = []
        for solid in self.solids: #Loop over solids
            facets = []
            for vi in range(len(solid)-1): #Loop over vertices in the solid
                facets.append(Facet(np.array(solid[vi]), np.array(solid[vi+1])))
            allFacets.append(facets)
        return allFacets

    def _getAllCutFacets(self):
        """
        Same as _getAllFacets, but creates a nested list of CutFacets.
        """
        allCutFacets = []
        for solidFacets in self.allFacets: #Loop over solids
            cutFacets = []
            for facet in solidFacets: #Loop over facets in solid
                cutFacets.append(facet.getCutFacets())
            allCutFacets.append(cutFacets)
        return allCutFacets
