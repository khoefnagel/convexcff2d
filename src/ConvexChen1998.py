"""
Module that implements a simplified version of the volumetric boundary
condition presented by Chen et al. (https://doi.org/10.1142/S0129183198001151).
Limitations:
-Only concave boundaries
-Only 2D
-Only no-slip (No moving/slip walls)

Reuses classes from CFF

Requires: numpy, scipy
"""

import numpy as np
import sympy as sym
from scipy import sparse

from Pgram import Pgram
from Geometry import Geometry


class ConvexChen1998(Geometry):
    """
    Class that implements a simplified version of the volumetric boundary
    condition presented by Chen et al. (https://doi.org/10.1142/S0129183198001151).

    Limitations:
    -Only concave boundaries
    -Only 2D
    -Only no-slip (No moving/slip walls)

    Attributes
    ----------
    velSetName : string
        Name of the velocity set used, e.g. "D2Q9".
    N : array_like
        Number of lattice nodes in each coordinate direction e.g. (Ny, Nx)
        for 2D, shape (D).
    solids : list of array_like
        List of the coordinates of the boundary of the solids, each entry
        should be an array of shape (Nv, D) where Nv is the number of
        boundary vertices.
    forceDens : None or array_like
        Body force density vector or None to not use a body force.
        The default is None.
    F : array_like
        Populations, shape (Q, Ny, Nx, 1).
    SM : sparse matrix
        Sparse streaming matrix including the effect of the Chen et al. boundary
        condition, shape (Q*Ny*Nx*1, Q*Ny*Nx*1). Stream is performed as F = SM @ F.
    S2fsurf : sparse matrix
        Matrix to compute the force on each facet from the input populations,
        shape (2*N_facets, Q*Nx*Ny) with N_facets the number of facets.
    S2Ftot : sparse matrix
        Matrix to compute the total force on each solid from the input populations,
        shape (2*N_solids, Q*Nx*Ny) with N_solids the number of solids.    
    """


    def __init__(self, velSetName, N, solids, forceDens=None):

        Geometry.__init__(self, velSetName, N, solids, 0, forceDens=forceDens)
        
        self._getPgrams()
        self._getMatrices()
        self._correctSolidStreaming()

    def _getPgrams(self):
        """
        Function to obtain self.allPgrams; a nested list that holds all the
        pgrams. The outer list corresponds to the solids, the next list to the
        uncut facets of that solid, the next list to the cut facets of that
        uncut facet. The innermost list contains the pgrams associated with
        that cut facet for each population direction i.
        """
        
        self.allPgrams = []

        self.nPgrams = 0
        for uncutFacets in self.allCutFacets: # Loop over solids
            solidPgrams = []
            for cutFacets in uncutFacets: # Loop over uncut facets in solid
                uncutFacetPgrams = []
                for facet in cutFacets: #Loop over cut facets in uncut facet
                    cutFacetPgrams = []
        
                
                    #Skip facets that lie on the boundary and point outwards
                    #This prevents periodicity problems
                    if facet.normal.dot([1, 0]) > 1 - 1e-10 and\
                         abs(facet.start[0] - self.N[1]) < 1e-10:
                        continue
                    elif facet.normal.dot([-1, 0]) > 1 - 1e-10 and\
                         abs(facet.start[0]) < 1e-10:
                        continue
                    elif facet.normal.dot([0, 1]) > 1 - 1e-10 and\
                         abs(facet.start[1] - self.N[0]) < 1e-10:
                        continue
                    elif facet.normal.dot([0, -1]) > 1 - 1e-10 and\
                         abs(facet.start[1]) < 1e-10:
                        continue

                    for i in range(self.Q):
                        Cj = self.Cij[i].squeeze()

                        # Only consider populations into the boundary
                        if np.dot(facet.normal, Cj) > 1e-14:
                            continue

                        # Create the pgram associated with the cut facet and Cj.
                        pgram = Pgram(facet, Cj)

                        # Store population index as a property of the pgram,
                        # this is needed later
                        pgram.i = i


                        # Skip facets outside the domain
                        if not 0 <= pgram.crossedCells[0][0] < self.N[0] or\
                           not 0 <= pgram.crossedCells[0][1] < self.N[1]:
                            continue

                        cutFacetPgrams.append(pgram)
                    uncutFacetPgrams.append(cutFacetPgrams)
                solidPgrams.append(uncutFacetPgrams)
            self.allPgrams.append(solidPgrams)
        

            # Total number of pgrams
            self.nPgrams = len(sum(sum(sum(self.allPgrams, []), []), []))

        return

    def _getMatrices(self):
        """
        Get the matrices for the streaming step. These are divided into the 
        various steps of Chen et al. (1998) to make the implementation clearer. 
        See the paper for details.      
        """

        # Matrix from f_i(x, t) to Gamma_i^{in, alpha)(t)
        self.Fi2Gamma_i_alpha = sparse.dok_matrix((self.nPgrams, np.prod(self.N)*self.Q))

        # Matrix from Gamma_i^{in, alpha}(t) to Q_i(x + c_i, t), the latter
        # represents the number of reflected particles N_i.
        self.Gamma_i_alpha2Qi = sparse.dok_matrix((np.prod(self.N)*self.Q, self.nPgrams))

        # Matrix from N_i(x, t) to N_i(x + c_i, t): direct streaming without reflections.
        # Mathematically it represents multiplication by (1 - P_i^s(x))
        self.Ni2NiDS = sparse.dok_matrix((np.prod(self.N)*self.Q,
                                                np.prod(self.N)*self.Q))

        # Matrices to compute forces
        self.Fi2fsurf = sparse.dok_matrix((self.D*len(sum(self.allFacets, [])),
                                           np.prod(self.N)*self.Q))
        self.Fi2Ftot = sparse.dok_matrix((self.D*len(self.allFacets),
                                          np.prod(self.N)*self.Q))

        # Matrix from f_i(x, t) to N_i(x, t)
        self.Fi2Ni = sparse.diags(np.tile(self.cellVolumes.flatten(), self.Q))

        # Matrix from N_i(x, t) to f_i(x, t)
        self.Ni2Fi = sparse.diags(np.tile(1/np.maximum(self.cellVolumes.flatten(), 1e-12), self.Q))
        
        # Ravel 3D index (q, yi, xi) to 1D
        ravel = lambda *ind : np.ravel_multi_index(ind, (self.Q, *self.N))

        # Initialize Ni2NiDS as 1 at the streaming indices; P_i^s(x) is later
        # subtracted.
        for i in range(self.Q):
            Cj = np.squeeze(self.Cij[i])
            for yi in range(self.N[0]):
                for xi in range(self.N[1]):
                    ci = np.array([i, yi, xi])
                    streamInd = np.array([i, *(ci[1:] + Cj[::-1])%self.N])
                    self.Ni2NiDS[ravel(*streamInd), ravel(*ci)] = 1

        # Loop over all pgrams to fill the matrices
        totFacetInd = -1
        pgramInd = -1
        for solidInd, (uncutFacets, uncutPgrams) in enumerate(zip(self.allFacets, self.allPgrams)):
            for (uncutFacet, uncutPgrams) in zip(uncutFacets, uncutPgrams):
                totFacetInd += 1
                for cutPgrams in uncutPgrams:
                    for pgram in cutPgrams:
                        pgramInd += 1

                        self._addPgram(pgram, totFacetInd, pgramInd,
                                              uncutFacet, solidInd)


        # Convert to csc format which is more efficient for matrix-matrix products
        self.Ni2NiDS = self.Ni2NiDS.asformat('csc')
        self.Fi2Gamma_i_alpha = self.Fi2Gamma_i_alpha.asformat('csc')
        self.Gamma_i_alpha2Qi = self.Gamma_i_alpha2Qi.asformat('csc')
        self.Fi2fsurf = self.Fi2fsurf.asformat('csc')
        self.Fi2Ftot = self.Fi2Ftot.asformat('csc')

        # Same as Ni2NiDS except with f_i(x, t) as input instead of N_i(x, t)
        self.Fi2NiDS = self.Ni2NiDS @ self.Fi2Ni

        # Matrix from f_i(x, t) to Q_i(x, t) (reflected N_i).
        self.Fi2Qi =  self.Gamma_i_alpha2Qi @ self.Fi2Gamma_i_alpha

        # Streaming matrix from pre-stream populations f_i(x, t) to post-stream populations
        # f_i(x + c_i, t + 1).
        self.SM = self.Ni2Fi @ (self.Fi2NiDS + self.Fi2Qi)

        return

    def _addPgram(self, pgram, totFacetInd, pgramInd, uncutFacet, solidInd):
        """
        Add the contribution of a single pgram to the various matrices.

        Parameters
        ----------
        pgram : Pgram
            The pgram to add.
        totFacetInd : int
            Index of the uncut facet to which the pgram belongs, counting
            all facets of all solids.
        pgramInd : int
            Index of the pgram, counting all pgrams of all facets of all solids.
        uncutFacet : Facet
            The uncut facet to which the pgram belongs.
        solidInd : int
            Index of the solid to which the pgram belongs, counting all solids.
        """


        # Ravel 3D index (q, yi, xi) to 1D
        ravel = lambda *ind: np.ravel_multi_index(ind, (self.Q, *self.N))

        i = pgram.i
        ibar = self.Ibar[i] # Index of the opposite direction

        if pgram.degenerate: #Skip degenerate pgrams
            return

        # Volume of pgram
        V_i_pgram_alpha = np.abs(pgram.Cj.dot(pgram.normal))*pgram.L
        
        for subZone in sum(pgram.subZones, []): # Loop over subzones

            #---------------------------------------------------------------
            # Index calculations

            #Mesh indices of the cell in which the subzone lies
            ci = subZone.originCellId%self.N

            # Raveled indices of subzone for incoming and outgoing populations
            # respectively.
            indIn, indOut = ravel(i, *ci), ravel(ibar, *ci)

            # Raveled index of cell to which subzone would stream directly
            # with no boundary.
            indDS = ravel(i, *((subZone.originCellId + pgram.Cj[::-1])%self.N))                   

            #---------------------------------------------------------------
            # Volume parameter calculations
            
            # Volume of the subZone
            V_i_pgram_alpha_x = subZone.getMomentCC(0, 0)

            # Subzone volume fraction wrt cell volume
            P_i_S = V_i_pgram_alpha_x/self.cellVolumes[0, *ci, 0]

            # Subzone volume wrt pgram volume
            P_i_alpha = V_i_pgram_alpha_x/V_i_pgram_alpha

            #---------------------------------------------------------------
            # Update matrices

            # Correct Ni2NiDS by removing the direct stream (the subzone
            # reflects instead).
            self.Ni2NiDS[indDS, indIn] -= P_i_S

            # See Chen1998
            self.Fi2Gamma_i_alpha[pgramInd, indIn] += V_i_pgram_alpha_x
            self.Gamma_i_alpha2Qi[indOut, pgramInd] += P_i_alpha

            #---------------------------------------------------------------
            # Calculate forces
            
            FSurf = 2*pgram.Cj[:,None]*V_i_pgram_alpha_x

            # Add the force contribution to the facet
            self.Fi2fsurf[2*totFacetInd:\
                          2*totFacetInd+self.D, indIn] += FSurf/uncutFacet.L

            # Add the force contribution to the total surface
            self.Fi2Ftot[2*solidInd:2*solidInd+self.D, indIn] += FSurf

        return


    def _correctSolidStreaming(self):
        """
        In the current implementation, solid cells don't get any populations
        streamed to them, leading to zero density. A temporary fix implemented here
        is to modify the streaming matrix SM such that they keep their density,
        but it is all concentrated in the still populations.
        """

        # Shape of input and output populations
        self.shape1 = (self.Q, *self.N, 1) 
        self.shape4 = (self.Q, *self.N, 1)

        # Functions to ravel multi-indices to single indices
        self._ravel4 = lambda *key : np.ravel_multi_index(key, self.shape4)
        self._ravel1 = lambda *key : np.ravel_multi_index(key, self.shape1)

        # Convert to lil format for easier modification
        self.SM = self.SM.asformat('lil')
        
        solidInds = np.array(np.where(self.fluidMoments[0,:,:,0,0,None] < 1e-10)).T
        for solidInd in solidInds:

            rowInd = self._ravel4(0, *solidInd)

            for i in range(self.Q):
                colInd = self._ravel1(i, *solidInd)

                self.SM[rowInd, colInd] = 1.

        # Convert back to csc format for efficiency
        self.SM = self.SM.asformat('csc')
        return        
