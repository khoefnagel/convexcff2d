"""
Module containing the Projection class, which implements the projection of
the near-wall populations onto a discontinuous, piecewise linear basis.
This is done through the Fi2PC matrix, which takes the populations vector as an
input and outputs the polynomial coefficients of the piecewise linear basis
in each cell.

Requires: numpy, scipy
"""

import numpy as np
import itertools as it
import scipy.sparse as sparse

from Geometry import Geometry
from Pgram import Pgram


class Projection(Geometry):
    """Implements the projection of the near-wall populations onto a piecewise linear basis.
    This is done through the Fi2PC matrix, which takes the populations vector as an
    input and outputs the polynomial coefficients of the piecewise linear basis in each cell.
    
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
        Degree of the polynomial basis functions. Currently, only deg=1 is implemented.
    forceDens : None or array_like
        Body force density vector or None to not use a body force.
        The default is None.
    weighing : None or string
        Weighing method to use for the projection. None for no weighing, 'volume' for
        weighing by the fluid volume in the neighboring cells, 'OBCNone' for taking only
        pgram-overlapped cells into account. The default is None.
    xy : bool
        Whether to include an xy-term in the polynomial basis. The default is False.
    shape1 : tuple
        Shape of the input populations array, (Q, Ny, Nx, 1) for 2D.
    shape3 : tuple
        Shape of the output polynomial coefficients array, (Q, Ny, Nx, deg+1, deg+1) for 2D.
    Fi2PC : sparse matrix
        Sparse matrix projecting from populations to polynomial coefficients,
        shape (Q*Ny*Nx*(deg+1)*(deg+1), Q*Ny*Nx*1).
    """

    def __init__(self, velSetName, N, solids, deg=1, forceDens=None, weighing=None, xy=False):

        assert deg == 1, "Currently, only deg=1 is implemented."

        Geometry.__init__(self, velSetName, N, solids, deg, forceDens=forceDens)

        #Define shapes of the input (the populations), the intermediate (Spline
        #coefficients) and the result (polynomial coefficients).
        self.shape1 = (self.Q, *self.N, 1) #Populations
        self.shape3 = (self.Q, *self.N, *[deg+1]*2) #Polynomial coefficients

        #Functions to convert from a multi-dimensional index to the 1D vector index
        self._ravel1 = lambda *key : np.ravel_multi_index(key, self.shape1)
        self._ravel3 = lambda *key : np.ravel_multi_index(key, self.shape3)

        self.pgramOverlappedCells = self._registerPgramOverlappedCells()

        # Matrix projecting from populations to polynomial coefficients
        self.Fi2PC = self._getFi2PC(weighing=weighing, xy=xy)
        return
        
    def _registerPgramOverlappedCells(self):
        """
        Register the cells that are overlapped by any pgram.
        Later, only these cells will be projected onto the polynomial basis to
        save computational cost.

        Returns
        -------
        pgramOverlappedCells : array_like
            Boolean array of shape (Q, Ny, Nx, 1) indicating whether a cell is
            overlapped by a pgram in the respective direction.
        """

        #Approach: initialize each cell as False in qualifyingCells, then loop over
        #the cutFacets and create Pgram instance for each facet.
        #The Pgram class can detect with which cells it overlaps, these
        #are updated to be True in qualifyingCells.
        pgramOverlappedCells = np.zeros((self.Q, *self.N, 1), dtype=bool)
        for i in range(self.Q):
            Cj = self.Cij[i].squeeze()
            ibar = self.Ibar[i]
            for facets in self.allCutFacets: #Loop over solids
                for cutFacet in sum(facets, []): #Loop over cut facets

                    #Skip facets where the streaming direction does not point into
                    #the solid.
                    if Cj.dot(cutFacet.normal) >= 0:
                        continue

                    #Create instance of Pgram class
                    pgram = Pgram(cutFacet, Cj)

                    #Set all directions in all crossed cells to qualifying
                    for crossedCellsIds in pgram.crossedCells:
                        if self.solidCells[0, *crossedCellsIds%self.N, 0]:
                            continue
                        pgramOverlappedCells[:, *crossedCellsIds%self.N, 0] = True
        return pgramOverlappedCells

    def _getFi2PC(self, weighing=None, xy=False):
        """
        Get the sparse matrix projecting from populations to discontinuous, piecewise polynomials.
        The populations vector should be of shape (Q*Ny*Nx*1) and the
        output polynomial coefficients vector will be of shape (Q*Ny*Nx*(deg+1)*(deg+1)).
        The projection is done using a least-squares approach, taking the
        neighboring cells into account. The weighing method can be specified
        as an argument.

        Parameters
        ----------
        weighing : None or string
            Weighing method to use for the projection. None for no weighing, 'volume' for
            weighing by the fluid volume in the neighboring cells, 'OBCNone' for taking only
            pgram-overlapped cells into account. The default is None.
        xy : bool
            Whether to include an xy-term in the polynomial basis. The default is False.
        
        Returns
        -------
        Fi2PC : scipy.sparse.dok_matrix
            Sparse matrix projecting from populations to polynomial coefficients.
        """

        Fi2PC = sparse.dok_matrix((np.prod(self.shape3), np.prod(self.shape1)))

        if self.deg == 0: # Piecewise constants => Fi2PC is identity matrix
            for ci in range(np.prod(self.shape1)):
                Fi2PC[ci, ci] = 1.
            return Fi2PC


        ctrds = self.centroidsLoc

        # Indices of the neighboring cells to take into account for the projection
        deltaInds = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1],
                              [0, 1], [1, -1], [1, 0], [1, 1]])
        
        # Initialize vector B in the least-squares problem min ||A*C - B||, with C the
        # piecewise polynomial coefficients to be found.
        # Each row corresponds to the difference between the populations in the
        # respective neighboring cell and the populations in the cell itself.
        # Since the populations are not known a-priori, B is actually a matrix
        # where each column corresponds to a neighboring cell (including the cell
        # itself, which corresponds to the first column). The shape of B is thus
        # (number of neighboring cells, number of neighboring cells + 1).
        B = np.hstack([-np.ones((len(deltaInds), 1)),
                       np.diag(np.ones(len(deltaInds)))])
        
        for i in range(self.Q):
            for ci in np.vstack(np.where(self.pgramOverlappedCells[i,:,:,0])).T:

                # Initialize matrix A in the least-squares problem min ||A*C - B||,
                # with C the piecewise polynomial coefficients to be found.
                # Each row corresponds to the difference in coordinates between the
                # centroid of a neighboring cell and the centroid of the cell itself.
                # The shape of A is thus (number of neighboring cells + 1, number of 
                # polynomial coefficients).
                A = []

                # Weights in the weighted least-squares problem
                weights = []

                # Valid indices of neighboring cells (i.e. not solid cells)
                validInds = []

                # Coordinates of the centroid of the cell itself
                xi, yi = ctrds[0, *ci, 0], ctrds[0, *ci, 1]

                for j, ciDelta in enumerate(deltaInds): # Loop over neighboring cells
                    
                    # Index of the neighboring cell
                    cj = (ci + ciDelta)%self.N

                    # Skip solid cells, these are not valid
                    if self.solidCells[0, *cj, 0]:
                        continue
                    validInds.append(j)
                    
                    # Coordinates of the centroid of the neighboring cell
                    xj = ctrds[0, *cj, 0] + ciDelta[1]
                    yj = ctrds[0, *cj, 1] + ciDelta[0]

                    if weighing is None: 
                        weights.append(1)
                    elif weighing == 'volume':
                        weights.append(self.fluidMoments[0, *cj, 0, 0])
                    elif weighing == 'OBCNone':
                        weights.append(self.pgramOverlappedCells[i, *cj, 0])
                    else:
                        raise Exception('Weighing not (yet) implemented.')

                    if xy:
                        A.append([xj - xi, yj - yi, xj*yj - xi*yi])
                    else:
                        A.append([xj - xi, yj - yi])

                A = np.array(A)
                W = np.diag(weights)

                # Solve the weighted least-squares problem to find the polynomial coefficients C       
                C = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ B[validInds]

                for j, ciDelta in enumerate([[0, 0], *deltaInds]): # Loop over neighboring cells + the cell itself

                    # Index of the neighboring cell
                    cj = (ci + ciDelta)%self.N

                    # Find the polynomial coefficients in fi = a + b*x + c*y (+ d*xy)
                    # with (x,y) the coordinates relative to the centroid of the cell
                    if xy:
                        b, c, d = C[:, j]
                        a = -b*xi - c*yi - d*xi*yi
                    else:
                        b, c = C[:, j]
                        a = -b*xi - c*yi
                    if j == 0:
                        a += 1
                    
                    # Add the entries to the sparse matrix Fi2PC
                    Fi2PC[self._ravel3(i, *ci, 0, 0), self._ravel1(i, *cj, 0)] += a
                    Fi2PC[self._ravel3(i, *ci, 1, 0), self._ravel1(i, *cj, 0)] += b
                    Fi2PC[self._ravel3(i, *ci, 0, 1), self._ravel1(i, *cj, 0)] += c
                    if xy:
                        Fi2PC[self._ravel3(i, *ci, 1, 1), self._ravel1(i, *cj, 0)] += d       

        # For non-pgram-overlapped cells, the input populations should be constant in the cell,
        # corresponding to an entry of 1 in Fi2PC for the zeroth polynomial coeff.
        for i in range(self.Q):
            for ci in it.product(range(self.N[0]), range(self.N[1])):
                if not self.pgramOverlappedCells[i, *ci, 0]:
                    Fi2PC[self._ravel3(i, *ci, 0, 0),
                         self._ravel1(i, *ci, 0)] = 1.
     
        return Fi2PC
    