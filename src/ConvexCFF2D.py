"""
Module containing the ConvexCFF2D class, which implements a 2D LBM solver which
implements the CFF boundary condition for convex, non-moving solids.

Requires: numpy, itertools
"""

import numpy as np
import itertools as it

from Projection import Projection
from ReflectAndIntegrate import ReflectAndIntegrate


class ConvexCFF2D(Projection, ReflectAndIntegrate):
    """
    Represents a 2D LBM solver which implements the CFF boundary condition for
    convex, non-moving solids.

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
    deg : integer
        Degree used for the splines. Default is 1.
    forceDens : None or array_like
        Body force density vector or None to not use a body force.
        The default is None.
    weighing : None or array_like
        Weighing factors for the projection step or None to not use weighing.
        The default is None.
    xy : bool
        Whether to include an xy-term in the polynomial basis. The default is False.
    F : array_like
        Populations, shape (Q, Ny, Nx, 1).
    SM : sparse matrix
        Sparse streaming matrix including the effect of the CFF boundary condition,
        shape (Q*Ny*Nx*1, Q*Ny*Nx*1). Stream is performed as F = SM @ F.
    Fi2fsurf : sparse matrix
        Matrix to compute the force on each facet from the input populations,
        shape (2*N_facets, Q*Nx*Ny) with N_facets the number of facets.
    Fi2Ftot : sparse matrix
        Matrix to compute the total force on each solid from the input populations,
        shape (2*N_solids, Q*Nx*Ny) with N_solids the number of solids.
    """

    def __init__(self, velSetName, N, solids, deg=1,
                 forceDens=None, weighing=None, xy=False):

        # Set up the Fi2PC matrix
        Projection.__init__(self, velSetName, N, solids, deg=deg,
                            forceDens=forceDens, weighing=weighing, xy=xy)
        
        # Set up the RI, PC2fsurf and PC2Ftot matrices
        ReflectAndIntegrate.__init__(self, velSetName, N, solids, deg=deg,
                                     forceDens=forceDens)

        # Set the streaming matrix SM
        self.SM = self.RI @ self.Fi2PC

        # Correct the streaming matrix for solid cells to prevent zero densities
        # in these cells.
        self._correctSolidStreaming()

        self.Fi2fsurf = self.PC2fsurf @ self.Fi2PC
        self.Fi2Ftot = self.PC2Ftot @ self.Fi2PC

        return


    def _correctSolidStreaming(self):
        """
        In the current implementation, solid cells don't get any populations
        streamed to them, leading to zero density. A temporary fix implemented here
        is to modify the streaming matrix SM such that they keep their density,
        but it is all concentrated in the still populations.
        """

        self.SM = self.SM.asformat('lil')
        
        solidInds = np.array(np.where(self.fluidMoments[0,:,:,0,0,None] < 1e-10)).T
        for solidInd in solidInds:
            ri = self._ravel4(0, *solidInd)

            for i in range(self.Q):
                ci = self._ravel1(i, *solidInd)

                self.SM[ri, ci] = 1.

        self.SM = self.SM.asformat('csc')

    def interpolatePPs(self, X, Y):
        """
        Interpolate the populations at the points (X, Y) using the piecewise
        polynomial coefficients.
        
        Parameters
        ----------
        X, Y : array_like
            Coordinates where the populations should be interpolated, shape
            (NyInterp, NxInterp). The coordinates should be in the range
            [0, Nx]x[0, Ny], where Nx and Ny are the number of cells in x- and
            y-direction respectively.
        
        Returns
        -------
        FInterp : array_like
            Interpolated populations at the points (X, Y), shape
            (Q, NyInterp, NxInterp).
        """
        FInterp = np.zeros((self.Q, *X.shape))

        PC = (self.Fi2PC @ self.F.flatten()).reshape(self.shape3)

        for i in range(self.Q):
            for ii in it.product(range(X.shape[0]), range(X.shape[1])):
                for pi in it.product(*[range(self.deg+1)]*2):
                    ci = (int(Y[*ii]//1), int(X[*ii]//1))
                    FInterp[i, *ii] += PC[i, *ci, *pi] * (X[*ii]%1 - 0.5)**pi[0] *\
                                       (Y[*ii]%1 - 0.5)**pi[1]
        return FInterp


