"""
Module containing the ReflectAndIntegrate class. This class implements the streaming
(including the geometric mapping at the boundary) of the discontinuous piecewise
polynomials and subsequent integration over the cells to obtain the post-stream
populations in each cell. Also implements the calculation of the forces on the boundary.

Requires: numpy, scipy, itertools
"""

import numpy as np
import scipy.sparse as sparse
import itertools as it

from Geometry import Geometry
from Pgram import Pgram


class ReflectAndIntegrate(Geometry):
    """
    Implements the streaming (including the geometric mapping at the boundary) of the 
    discontinuous piecewise polynomials and subsequent integration over the cells to obtain
    the post-stream populations in each cell. Both streaming and integration is implemented in the
    matrix RI, which takes the vector of polynomial spline coefficients in each cell
    as input and outputs the post-stream populations in each cell.

    Attributes:
    ----------
    velSetName : string
        Name of the velocity set to be used, e.g. "D2Q9".
    N : array_like
        Number of lattice nodes in each coordinate direction e.g. (Ny, Nx)
        for 2D, shape (D).
    solids : list of array_like
        List of the coordinates of the boundary of the solids, each entry
        should be an array of shape (Nv, D) where Nv is the number of boundary vertices.
    deg : integer
        Degree to use for the splines.
    forceDens : None or array_like
        Body force density vector or None to not use a body force.
        The default is None.
    RI : sparse matrix
        Reflect and integrate matrix, shape (Q*Nx*Ny, Q*Nx*Ny*(deg+1)**2)
        with Q the number of populations, Nx and Ny the number of cells in x- and
        y-direction respectively, and deg the degree of the splines.
    PC2fsurf : sparse matrix
        Matrix to compute the force on each facet from the piecewise polynomial coefficients,
        shape (2*N_facets, Q*Nx*Ny*(deg+1)**2) with N_facets the number of facets.
    PC2Ftot : sparse matrix
        Matrix to compute the total force on each solid from the piecewise polynomial coefficients,
        shape (2*N_solids, Q*Nx*Ny*(deg+1)**2) with N_solids the number of solids.
    """

    def __init__(self, velSetName, N, solids, deg=1, forceDens=None):

        Geometry.__init__(self, velSetName, N, solids, deg, forceDens=forceDens)

        #Define the shapes of the input (polynomial coefficients) and the output
        #(the populations).
        self.shape3 = (self.Q, *self.N, *[deg+1]*2)
        self.shape4 = (self.Q, *self.N, 1)
        
        #Functions to convert from a multi-dimensional index to the 1D vector index
        self._ravel3 = lambda *key : np.ravel_multi_index(key, self.shape3)
        self._ravel4 = lambda *key : np.ravel_multi_index(key, self.shape4)
       
        # Construct the RI matrix as well as the force matrices PC2fsurf and PC2Ftot
        self._getRIMatrix()
        return        

    def _getRIMatrix(self):
        """
        Construct the reflect and integrate matrix RI as well as the force matrices
        PC2fsurf and PC2Ftot.
        """

        # Initialize as sparse DOK matrix for efficient incremental construction
        self.RI = sparse.dok_matrix((np.prod(self.shape4), np.prod(self.shape3)))


        self.PC2fsurf = sparse.dok_matrix((self.D*len(sum(self.allFacets, [])),
                                           np.prod(self.shape3)))
        self.PC2Ftot = sparse.dok_matrix((self.D*len(self.allFacets),
                                          np.prod(self.shape3)))

        # Step 1: Naive streaming of the fluid portion of each cell
        self._fluidStream()

        # Step 2: Add the effect of the boundaries by removing the stream through
        # the boundary and adding the reflection off the boundary.
        # This is also where the forces on the boundary are calculated.
        self._addReflectionAndForce()

        # Step 3: Convert the entries in RI from number of particles Ni to 
        # particle density fi by dividing by the fluid cell volume
        self._correctVolume()

        # Convert to csc format which is more efficient for matrix-matrix products
        self.RI = self.RI.asformat('csc')

        return

    def _fluidStream(self):
        """
        Encode naive periodic streaming of the fluid portion of each cell into RI;
        boundaries are not taken into account, so the fluid will stream straight
        through them rather than reflecting. This function does take into account
        the piecewise polynomials.
        """

        # Output indices post-stream in 1D
        indsOutPost1D = np.arange(self.Q*self.N[0]*self.N[1])

        # Output indices post-stream in same shape as F (4D)
        indsOutPost4D = indsOutPost1D.reshape(self.Q, *self.N, 1)

        # Stream output indices backwards to find corresponding pre-stream index
        indsOutPre4D = np.stack([np.roll(indsOutPost4D[i], -self.Cij[i].squeeze(),
                                    tuple(range(self.D-1,-1,-1)))\
                                for i in range(self.Q)])

        # Convert pre-stream indices to 1D: these will be the column indices in RI
        indsOutPre1D = (indsOutPre4D*np.ones((self.deg+1)**2)).reshape(-1)

        # Input indices in 1D; simply an arange: these will be the row indices in RI
        indsIn1D = np.arange(self.Q*self.N[0]*self.N[1]*(self.deg+1)**2)

        # Fluid moments flattened to 1D list of right shape
        moments1D = (self.fluidMoments[:,:,:,:self.deg+1,:self.deg+1]*\
                     np.ones((self.Q, 1, 1, 1, 1))).reshape(-1)

        self.RI[indsOutPre1D, indsIn1D] = moments1D

        return

    def _addReflectionAndForce(self):
        """
        Add the effect of the boundaries to RI, this involves two steps:
        -Removing the stream through the boundary
        -Adding the reflection on the boundary

        This function also computes the forces on the facets and encodes
        them in the matrices PC2fsurf and PC2Ftot.
        """

        totFacetInd = -1 #Index of the current facet over all solids
        for solidInd, (uncutFacets, cutFacets) in\
                            enumerate(zip(self.allFacets, self.allCutFacets)):
            for uncutFacet, facetCutFacets in zip(uncutFacets, cutFacets):
                totFacetInd += 1
                for cutFacet in facetCutFacets:

                    # Skip facets that lie on the domain edge and point outwards
                    # This prevents periodicity problems
                    if cutFacet.normal.dot([1, 0]) > 1 - 1e-10 and\
                       abs(cutFacet.start[0] - self.N[1]) < 1e-10:
                        continue
                    elif cutFacet.normal.dot([-1, 0]) > 1 - 1e-10 and\
                         abs(cutFacet.start[0]) < 1e-10:
                        continue
                    elif cutFacet.normal.dot([0, 1]) > 1 - 1e-10 and\
                         abs(cutFacet.start[1] - self.N[0]) < 1e-10:
                        continue
                    elif cutFacet.normal.dot([0, -1]) > 1 - 1e-10 and\
                         abs(cutFacet.start[1]) < 1e-10:
                        continue

                    self._addReflectionAndForceCutFacet(uncutFacet, cutFacet, solidInd, totFacetInd)
        return


    def _addReflectionAndForceCutFacet(self, uncutFacet, cutFacet, solidInd, totFacetInd):
        """
        Add the effect of a single cut facet to RI, this involves two steps:
        -Removing the stream through the boundary
        -Adding the reflection on the boundary

        This function also computes the forces on the facet and encodes
        them in the matrices PC2fsurf and PC2Ftot.

        Parameters:
        ----------
        uncutFacet : Facet
            The uncut facet of which cutFacet is a part.
        cutFacet : CutFacet
            The cut facet to consider.
        solidInd : integer
            Index of the solid to which the facet belongs.
        totFacetInd : integer
            Index of the facet over all solids.
        """

        for i in range(self.Q):
            
            Cj = self.Cij[i].squeeze()
            ibar = self.Ibar[i] # Index of the population opposite to i

            # Only consider populations traveling into the boundary
            if np.dot(cutFacet.normal, Cj) > -1e-14:
                continue

            # Create the pgram associated with the cut boundary facet and Cj
            pgram = Pgram(cutFacet, Cj)

            # Skip facets outside the domain
            if not 0 <= pgram.crossedCells[0][0] < self.N[0] or\
               not 0 <= pgram.crossedCells[0][1] < self.N[1]:
                continue

            # Consider each piecewise polynomial in each subzone
            for degx, degy in it.product(*[range(self.deg+1)]*2):
                for subZone in sum(pgram.subZones, []):

                    # Get the moment over the subzone for the (degx, degy)-th polynomial term
                    subZoneMoment = subZone.getMomentCC(degx, degy)

                    # Get the column index colInd
                    origInd = subZone.originCellId%self.N
                    colInd = self._ravel3(i, *origInd, degx, degy)
                    
                    # Get rmRowInd: the row index of the cell where the stream should be removed
                    # since it streams through the boundary.
                    rmDestInd = (origInd + Cj[::-1])%self.N
                    rmRowInd = self._ravel4(i, *rmDestInd, 0)
                    self.RI[rmRowInd, colInd] -= subZoneMoment

                    # Get rflRowInd: the row index of the cell where to add the reflection 
                    # on the boundary.
                    rflDestInd = subZone.destCellId%self.N
                    rflRowInd = self._ravel4(ibar, *rflDestInd, 0)
                    self.RI[rflRowInd, colInd] += subZoneMoment

                    #======================================================
                    # Forces

                    # Force on the facet
                    FSurf = 2*Cj[:,None]*subZoneMoment

                    # Add the force contribution to the facet. Divide by the
                    # length of the uncut facet to get force density.
                    self.PC2fsurf[2*totFacetInd:\
                                  2*totFacetInd+self.D, colInd] +=\
                                                            FSurf/uncutFacet.L

                    # Add the force contribution to the total force on the solid.
                    self.PC2Ftot[2*solidInd:2*solidInd+self.D, colInd] += FSurf
        return        

    def _correctVolume(self):
        """
        Since RI integrates the populations over the fluid portion of each cell,
        the entries in RI are currently the number of particles Ni in the fluid portion
        of the cell. To convert this to the particle density fi, the entries in RI
        need to be divided by the cell fluid volume. See Chen et al. 
        (https://doi.org/10.1142/S0129183198001151) for details.
        """

        # Fluid volumes
        V = self.fluidMoments[:,:,:,0,0,None]*np.ones((self.Q, 1, 1, 1))
        divVDiag = sparse.diags((1/np.maximum(V, 1e-12)).reshape(-1))
        self.RI = divVDiag @ self.RI

        return
    