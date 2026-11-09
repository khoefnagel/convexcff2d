"""
LBMBase.py

Base class for LBM, can be used for fully periodic problems.

Requires: numpy
"""

import numpy as np


class VelSet:
    """
    Simple container class for velocity sets.

    Attributes:
    ----------
    D : scalar
        Dimension
    Q : scalar
        Number of populations
    Wi : array_like
        Weights, shape (Q, 1, ..., 1)
    Cij : array_like
        Population vectors, shape (Q, 1, ..., D)
    """

    def __init__(self, D: int, Q: int, Wi: np.ndarray, Cij: np.ndarray):
        
        self.D = D
        self.Q = Q
        self.Wi = np.array(Wi).reshape((Q, *[1]*(D+1)))
        self.Cij = np.array(Cij).reshape((Q, *[1]*D, D))

        return

#Initialize standard velocity sets
D1Q3 = VelSet(1, 3, [2/3, 1/6, 1/6], [[0], [1], [-1]])

D2Q9 = VelSet(2, 9, [4/9, *[1/9]*4, *[1/36]*4],
              [[0, 0],
               [1, 0], [0, 1], [-1, 0], [0, -1],
               [1, 1], [-1, 1], [-1, -1], [1, -1]])

D3Q19 = VelSet(3, 19, [1/3, *[1/18]*6, *[1/36]*12],
               [[0, 0, 0], #Still
                [1, 0, 0], [0, 1, 0], [0, 0, 1], #Mag 1
                [-1, 0, 0], [0, -1, 0], [0, 0, -1], #Mag 1
                [1, 1, 0], [1, 0, 1], [0, 1, 1], #Mag sqrt(2)
                [-1, 1, 0], [-1, 0, 1], [0, -1, 1], #Mag sqrt(2)
                [1, -1, 0], [1, 0, -1], [0, 1, -1], #Mag sqrt(2)
                [-1, -1, 0], [-1, 0, -1], [0, -1, -1] #Mag sqrt(2)
                ])


class LBMBase:
    """
    Base class for LBM, can be used for fully periodic problems.

    Attributes
    ----------
    D, Q, Wi, Cij : See velSet class
    N : array_like,
         Number of lattice nodes in each coordinate direction e.g. (Nz, Ny, Nx)
         for 3D, shape (D).
    F : array_like,
        Populations at the current step, shape (Q, *N, 1).
    ibar : array_like
        Population index opposite to population index i. For example, if i
        corresponds to C_i=[1, -1] then ibar corresponds to C_{ibar}=[-1, 1].
        Here, ibar is a list corresponding to range(Q) as input i.
    forceDens : array_like,
        Body force density, shape (1, *N, D). Default is None.
    bodyForce : bool,
        Whether there is a body force present. Default is False.

    Methods
    -------
    fEq(rho, U)
        Get the equilibrium distribution function for a given density and velocity.
    initialize(rho, U)
        Initialize the populations at equilibrium corresponding to specified density and velocity.
    getMoments()
        Get the moments (density and velocity) based on the current populations.
    collide(tau)
        Perform the collision step with the BGK collision model.
    stream()
        Perform the streaming step using the sparse streaming matrix SM.
    streamPeriodic()
        Perform the streaming step using numpy roll, assuming periodic boundaries.   
    """

    def __init__(self, velSetName: str, N: tuple, forceDens=None):

        assert isinstance(velSetName, str), 'velSetName should be a string'
        assert velSetName in globals(), f'Velocity set {velSetName} not defined'
        assert len(N) == globals()[velSetName].D, f'Length of N should be {globals()[velSetName].D}'

        velSet = globals()[velSetName]

        self.D = velSet.D
        self.Q = velSet.Q
        self.Wi = velSet.Wi
        self.Cij = velSet.Cij

        self.N = np.array(N).flatten()
        self.F = np.zeros((self.Q, *self.N, 1))

        self.Ibar = self._getIbar()

        if forceDens is None:
            self.bodyForce = False
        else:
            self.bodyForce = True
            self.forceDens = forceDens.reshape((*[1]*(self.D+1), self.D))

            if self.D == 2:
                self.forceTermConst, self.forceTermUx, self.forceTermUy = self._precomputeForceTermComps2D()
            elif self.D == 3:
                raise NotImplementedError('Body force not implemented for D=3 yet')

        return

    def _getIbar(self):
        """
        Get the list of opposite population indices. I.e. Cij[ibar[i]]=-Cij[i].

        Returns
        -------
        ibar : array_like
            List of opposite population indices, shape (Q,).
        """

        #Find for each Cj the closest antiparallel vector; the corresponding i
        #is ibar.
        Ibar = np.where(np.min(np.isclose(self.Cij.reshape(self.Q,self.D),
                        -self.Cij.reshape(self.Q,1,self.D)), axis=-1))[1]
        return Ibar

    def fEq(self, rho, U):
        """
        Get the equilibrium distribution function for a given density and velocity.

        Parameters
        ----------
        rho : array_like
            Density, shape (1, *N, 1).
        U : array_like
            Velocity, shape (1, *N, D).
        
        Returns
        -------
        F : array_like
            Equilibrium populations, shape (Q, *N, 1).
        """

        #Precompute U dot Cij and U dot U
        UdotCij = np.sum(U*self.Cij, axis=-1, keepdims=True)
        UdotU = np.sum(U*U, axis=-1, keepdims=True)

        #Return fEq, equation simplified using that Cs^2=1/3 in lattice units
        return self.Wi*rho*(1 + 3*UdotCij + 4.5*(UdotCij*UdotCij) - 1.5*UdotU)
        
    def initialize(self, rho, U):
        """
        Initialize the populations at equilibrium corresponding to specified density and velocity.
        
        Inputs:
        rho : array_like
            Density, shape (1, *N, 1).
        U : array_like
            Velocity, shape (1, *N, D). 
        """

        if not self.bodyForce:
            self.F = self.fEq(rho, U)

        # If there is a body force, the equilibrium is shifted by 0.5*forceDens/rho
        else:
            self.F = self.fEq(rho, U-0.5*self.forceDens/rho)   
        return

    def getMoments(self):
        """
        Get the moments (density and velocity) based on the current populations.

        Returns
        -------
        rho : array_like
            Density, shape (1, *N, 1).
        U : array_like
            Velocity, shape (1, *N, D).
        """

        #See any LBM book for these relations
        rho = np.sum(self.F, axis=0, keepdims=True)

        if not self.bodyForce:
            U = np.sum(self.F*self.Cij, axis=0, keepdims=True)/rho

        # If there is a body force, the velocity is shifted by 0.5*forceDens/rho
        else:
            U = (np.sum(self.F*self.Cij, axis=0, keepdims=True) +
                 0.5*self.forceDens)/rho
            
        return rho, U

    def _precomputeForceTermComps2D(self):
        """
        Precompute components of the force term to correct the collision step, see Ch. 6 of the Kruger book.
        This is done to speed up the computation of the force term during each collision step.

        Returns
        -------
        forceTermConst: array_like
            Constant term in the force term, shape (Q, *N, 1).
        forceTermUx: array_like
            Coefficient of Ux in the force term, shape (Q, *N, 1).
        forceTermUy: array_like
            Coefficient of Uy in the force term, shape (Q, *N, 1).
        """
        Cx = self.Cij[:,:,:,0,None]
        Cy = self.Cij[:,:,:,1,None]        
        Fx = self.forceDens[:,:,:,0]
        Fy = self.forceDens[:,:,:,1]

        forceTermConst = 3*self.Wi*(Cx*Fx + Cy*Fy)
        forceTermUx = 9*self.Wi*((Cx**2 - 1/3)*Fx + (Cx*Cy)*Fy)
        forceTermUy = 9*self.Wi*((Cx*Cy)*Fx + (Cy**2 - 1/3)*Fy)

        return forceTermConst, forceTermUx, forceTermUy

        
    def _forceTerm(self, U):
        """
        Compute the force term for the given velocity field. See Ch. 6 of the Kruger book.
        Reuses precomputed components to speed up the computation.

        Parameters
        ----------
        U : array_like
            Velocity, shape (1, *N, D).
        
        Returns
        -------
        forceTerm : array_like
            Force term to correct the collision step, shape (Q, *N, 1).
        """

        Ux = U[:,:,:,0,None]
        Uy = U[:,:,:,1,None]

        return self.forceTermConst + self.forceTermUx*Ux + self.forceTermUy*Uy

    def collide(self, tau):
        """
        Perform the collision step with the BGK collision model.

        Parameters
        ----------
        tau : scalar
            Relaxation time in lattice units.
        """

        rho, U = self.getMoments()

        if not self.bodyForce:
            self.F -= (self.F - self.fEq(rho, U))/tau
        else: # There is a body force, see Ch. 6 of the Kruger book
            self.F *= (1 - 1/tau)
            self.F += self.fEq(rho, U)*(1/tau) + (1 - 0.5/tau)*self._forceTerm(U)
        return

    def stream(self):
        """
        Perform the streaming step using the sparse streaming matrix SM.
        """
        self.F = (self.SM @ self.F.flatten()).reshape((self.Q, *self.N, 1))
        return

    def streamPeriodic(self):
        """
        Perform the streaming step using numpy roll, assuming periodic boundaries.
        """
        self.F = np.stack([np.roll(self.F[i], self.Cij[i].squeeze(),
                                    tuple(range(self.D-1,-1,-1)))\
                            for i in range(self.Q)])
        return