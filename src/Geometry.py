"""
Module containing the Geometry class, which handles the geometry related operations
for the CFF boundary condition. This includes calculating up to the first order
moments of the fluid and solid in each cell.

Requires: numpy, itertools, shapely, math
"""

import numpy as np
import itertools as it
from shapely import box, Polygon
from math import comb

from LBMBase import LBMBase
from Boundary import Facets


def globToLoc(xg, yg, Cj):
    """
    Function to convert from global coordinates to local coordinates.
    Local coordinates share the same origin, but are rotated antiparallel to Cj.
    Furthermore, the local coordinate system is stretched by a factor ||Cj||.
    
    Parameters
    ----------
    xg : float
        x-coordinate in global coordinate system
    yg : float
        y-coordinate in global coordinate system
    Cj : array_like
        Velocity vector of the populations
    
    Returns
    -------
    xl : float
        x-coordinate in local coordinate system
    yl : float
        y-coordinate in local coordinate system
    """

    if np.linalg.norm(Cj) < 1e-10: # Still population
        return np.array([xg, yg])

    # Convert to local coordinates
    Cx, Cy = Cj.flatten()
    norm = np.linalg.norm(Cj)
    xl = (-Cy*xg + Cx*yg)/norm**2
    yl = (-Cx*xg - Cy*yg)/norm**2

    return np.array([xl, yl])

def locToGlob(xl, yl, Cj):
    """
    Function to convert from local coordinates back to global coordinates, see
    globToLoc for more details on the local coordinate system.

    Parameters
    ----------
    xl : float
        x-coordinate in local coordinate system
    yl : float
        y-coordinate in local coordinate system
    Cj : array_like
        Velocity vector of the populations

    Returns
    -------
    xg : float
        x-coordinate in global coordinate system
    yg : float
        y-coordinate in global coordinate system
    """

    if np.linalg.norm(Cj) < 1e-10: #Nonmoving populations
        return np.array([xl, yl])
    
    # Convert to global coordinates
    Cx, Cy = Cj.flatten()
    xg = (-Cy*xl - Cx*yl)
    yg = (Cx*xl - Cy*yl)

    return np.array([xg, yg])


def polygonMoment(polygon, degx, degy):
    """
    Analytically find integral of a polynomial term (e.g. x^2*y^4) over a polygon.
    Based on the following article:
    https://api.semanticscholar.org/CorpusID:17506973

    Parameters
    ----------
    polygon : array_like
        Array of shape (Nv, 2) with Nv the number of vertices in the polygon.
        The first column corresponds to x-coordinates, the second to y-coordinates.
        The first and last vertex should be the same to close the polygon.
    degx : int
        Degree of x in the polynomial term (x^degx).
    degy : int
        Degree of y in the polynomial term (y^degy).
    """

    #Zeroth moment over the polygon; the final moment is multiplied by the sign of
    #the zero moment to get the correct solution also for clockwise defined
    #polygons.
    zeroMoment = 0.5*sum([polygon[i-1][0]*polygon[i][1] - polygon[i][0]*polygon[i-1][1] for i in range(1, len(polygon))])
    correctionSign = np.sign(zeroMoment)
    
    p, q = degx, degy 
    
    leadingFrac = 1/((p + q + 2)*(p+q+1)*comb(p+q, p))
    isum = 0.
    for i in range(1, len(polygon)):
        xi, yi = polygon[i]
        xim1, yim1 = polygon[i-1]
        pqSum = 0.
        for k in range(p+1):
            for l in range(q+1):
                pqSum += comb(k+l, l)*comb(p+q-k-l, q-l)*\
                                                 xi**k*xim1**(p-k)*yi**l*yim1**(q-l)
        isum += (xim1*yi-xi*yim1)*pqSum            
    
    return leadingFrac*isum*correctionSign


class Geometry(LBMBase, Facets):
    """
    Class handling the geometry related operations for the CFF boundary condition.
    
    Attributes:
    ----------
    velSetName : string
        Name of the velocity set to be used, e.g. "D2Q9".
    N : array_like
        Number of lattice nodes in each coordinate direction e.g. (Ny, Nx)
        for 2D, shape (D).
    solids : list of array_like
        List of the coordinates of the boundary of the solids, each entry
        should be an array of shape (Nv, D) where Nv is the number of
        boundary vertices.
    deg : integer
        Degree to use for the splines.
    forceDens: None or array_like
        Body force density vector or None to not use a body force.
        The default is None.

    Attributes
    ----------
    solidMoments : array_like
        Array of shape (1, *N, deg+1, deg+1) containing the moments of the solid
        part of each cell up to the specified degree. The moments are defined
        with respect to the center of the cell.
    fluidMoments : array_like
        Array of shape (1, *N, deg+1, deg+1) containing the moments of the fluid
        part of each cell up to the specified degree. The moments are defined
        with respect to the center of the cell.
    cellVolumes : array_like
        Array of shape (1, *N, 1, 1) containing the volume of the fluid part of each cell.
    centroidsLoc : array_like
        Array of shape (1, *N, 2) containing the locations of the centroids of the fluid part of each cell.
        This is in local cell coordinates, i.e. with respect to the center of the cell.
    centroidsGlob : array_like
        Array of shape (1, *N, 2) containing the locations of the centroids of the fluid part of each cell.
        This is in global coordinates, i.e. with respect to the global origin.
    solidCells : array_like
        Boolean array of shape (1, *N, 1, 1) indicating whether a cell is completely solid.
    """

    def __init__(self, velSetName, N, solids, deg, forceDens=None):
        
        #Initialize parent classes
        LBMBase.__init__(self, velSetName, N, forceDens=forceDens)
        Facets.__init__(self, solids)

        self.deg = deg

        self.solidMoments, self.fluidMoments, self.cellVolumes = self._getSolidAndFluidMoments()
        self.centroidsLoc, self.centroidsGlob = self._getCentroids()
        self.solidCells = self._registerSolidCells()


    def _getSolidAndFluidMoments(self):
        """
        Function to obtain the moments of the fluid part as well as the solid part
        of each cell. These moments are defined with respect to the middle of the
        respective cell.
        
        Returns
        -------
        solidMoments : array_like
            Array of shape (1, *N, deg+1, deg+1) containing the moments of the solid
            part of each cell up to the specified degree. The moments are defined
            with respect to the center of the cell.
        fluidMoments : array_like
            Array of shape (1, *N, deg+1, deg+1) containing the moments of the fluid
            part of each cell up to the specified degree. The moments are defined
            with respect to the center of the cell.
        cellVolumes : array_like
            Array of shape (1, *N, 1, 1) containing the volume of the fluid part of each cell.
        """

        # Minimum degree is 1 such centroids can be found.
        deg = max(self.deg, 1)
        
        solidMoments = np.zeros((1, *self.N, *[deg+1]*2)) #Initialize at 0

        # Evaluate the moments of the unit cell from -0.5 to 0.5
        cellPoly = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5],
                             [-0.5, 0.5], [-0.5, -0.5]])
        cellMoments = np.zeros([deg+1]*2)
        for degx, degy in it.product(*[range(deg+1)]*2):
            cellMoments[degx, degy] = polygonMoment(cellPoly, degx, degy)

        # Initialize as if the whole cell is fluid, solid moments are
        # later subtracted.
        fluidMoments = np.tile(cellMoments, (1, *self.N, 1, 1))

        for solid in self.solids:
            solidPoly = Polygon(solid) # Solid as shapely.Polygon object

            # Optimization: only check cells within the rectangle around the solid
            bounds = np.array([*np.floor(solidPoly.bounds[:2]),
                               *np.ceil(solidPoly.bounds[2:])]).astype(int)

            # ci: y-cell index; cj: x-cell index
            for cj, ci in it.product(range(bounds[0], bounds[2]+1),
                                     range(bounds[1], bounds[3]+1)):

                # Make sure that [ci, cj] is inside the domain
                if not 0 <= ci < self.N[0] or not 0 <= cj < self.N[1]:
                    continue

                cell = box(cj, ci, cj+1, ci+1) # Represent cell as shapely.box object

                # Overlap between the solid and the cell
                overlap = solidPoly.intersection(cell)

                # If the overlap is empty or is e.g. a point, the solid volume of
                # the cell is zero and the code moves on to the next cell.
                if not isinstance(overlap, Polygon) or overlap.is_empty:
                    continue

                # Shift to cell centered coordinates
                overlapCC = np.array(overlap.exterior.coords[:]) -\
                            np.array([cj+0.5, ci+0.5])

                # Find the moments of the solid portion (overlap) up to the required
                # degree. Add to solid moments, subtract from fluid moments.
                for degx, degy in it.product(*[range(deg+1)]*2):
                    moment = polygonMoment(overlapCC, degx, degy)
                    solidMoments[0, ci, cj, degx, degy] += moment
                    fluidMoments[0, ci, cj, degx, degy] -= moment
        
        # Cell volumes are zeroth order fluid moments
        cellVolumes = fluidMoments[:,:,:,0,0,None]
        return solidMoments, fluidMoments, cellVolumes

    
    def _getCentroids(self):
        """
        Function to obtain the centroids of the fluid part of each cell, both in
        local and global coordinates.

        Returns
        -------
        centroidsLoc : array_like
            Array of shape (1, *N, 2) containing the locations of the centroids of the fluid part of each cell.
            This is in local cell coordinates, i.e. with respect to the center of the cell.
        centroidsGlob : array_like
            Array of shape (1, *N, 2) containing the locations of the centroids of the fluid part of each cell.
            This is in global coordinates, i.e. with respect to the global origin.
        """


        V = self.cellVolumes
        xcentroids = self.fluidMoments[:,:,:,1,0,None]/np.maximum(V, 1e-10)
        ycentroids = self.fluidMoments[:,:,:,0,1,None]/np.maximum(V, 1e-10)
        centroidsLoc = np.concatenate((xcentroids, ycentroids), axis=-1)

        centroidsGlob = np.array(centroidsLoc)
        centroidsGlob[0,:,:,0] += np.arange(self.N[1]) + 0.5
        centroidsGlob[0,:,:,1] += np.arange(self.N[0])[:, None] + 0.5    
        centroidsGlob = centroidsGlob
        
        return centroidsLoc, centroidsGlob

    def _registerSolidCells(self):
        """
        Function to register which cells are completely solid.
        
        Returns
        -------
        solidCells : array_like
            Boolean array of shape (1, *N, 1, 1) indicating whether a cell is completely solid.
        """
        return self.solidMoments[:,:,:,0,0,None] > 1 - 1e-10
