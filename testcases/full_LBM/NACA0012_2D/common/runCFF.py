"""
Run different boundary treatments for 2D flow over a NACA 0012 airfoil at Re_c=100. 
The following case parameters should be defined in the file ../caseDef.json:
- width : Number of cells in the x direction
- height : Number of cells in the y direction
- alphaDeg : Angle of attack of the airfoil in degrees
- c : Chord length of the airfoil in number of cells
- rf : Refinement factor compared to the base grid
- U : Inflow velocity in lattice units
- tau : Relaxation time
- Nt : Number of timesteps to run

The resulting lift and drag coefficients are saved in files <case>_Cl.npy/<case>_Cd.npy.
Furthermore, the pressure and skin friction coefficients on the airfoil surface
are saved in <case>_Cp.npy and <case>_Cf.npy, respectively.
The time convergence of the lift coefficient is plotted and saved
as ../timeConvergence/<case>_timeConvergence.pdf

Requires: numpy, matplotlib, json, sys
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '../../../../../../../src/')
from ConvexChen1998 import ConvexChen1998
from ConvexCFF2D import ConvexCFF2D

# Cases to run
runCases = ['Chen', 'CFFVol']


# Load variables defined in the caseDef json file into local namespace
with open('../caseDef.json') as f:
    locals().update(json.load(f))


 #Compute the actual chord Reynolds number and print it for verification
Re_c = (3*U*c)/(tau - 1/2)
print(f'Simulated chord Reynolds number: Re_c = {Re_c:.01f}')


#================================================================================
# Reading and processing airfoil coordinates

# Airfoil coordinate file
airfoilFile = './NACA0012.dat'

# Read in and process airfoil coords to form closed shape
# MODIFY IF NECESSARY!!
airfoilRaw = np.genfromtxt(airfoilFile, skip_header=1)
airfoilCoords = np.vstack((airfoilRaw, airfoilRaw[0]))

# Scale, translate, rotate airfoil coords
alpha = np.deg2rad(alphaDeg)
rotationMatrix = np.array([[np.cos(alpha), np.sin(alpha)],
                           [-np.sin(alpha), np.cos(alpha)]])
airfoilCoordsScaled = (airfoilCoords + np.array([-0.25, 0]))*c
airfoilCoordsRotated = np.einsum('ik,jk->ij', airfoilCoordsScaled, rotationMatrix)
airfoilCoordsTrans = airfoilCoordsRotated + np.array([width/4, height/2])

# List of solid objects in the flow (in this case only the airfoil)    
objects = [airfoilCoordsTrans]

# Compute normal and tangent vectors of the airfoil panels
dxdyAirfoil = np.diff(airfoilCoordsTrans, axis=0)
lAirfoil = np.linalg.norm(dxdyAirfoil, axis=1, keepdims=True)
tangentsAirfoil = dxdyAirfoil/lAirfoil
normalsAirfoil = dxdyAirfoil[:,::-1]*np.array([-1, 1])/lAirfoil

# Local (untransformed coordinates)
airfoilCoordsLocal = (airfoilCoords[:-1] + airfoilCoords[1:])/2

# Save untransformed and transformed airfoil coordinates separately
np.save('airfoilCoordsLocal', airfoilCoordsLocal)
np.save('airfoilCoordsGlobal', airfoilCoordsTrans)


#================================================================================
# Predefine cases

# Cases are defined as functions such that instances are only created later
# when the function is case called.
cases = {
    'Chen' : lambda : ConvexChen1998('D2Q9', (height, width), objects),
    'CFFConst' : lambda : ConvexCFF2D('D2Q9', (height, width), objects, 0),
    'CFFNone' : lambda : ConvexCFF2D('D2Q9', (height, width), objects, 1,
                                      weighing=None),    
    'CFFVol' : lambda : ConvexCFF2D('D2Q9', (height, width), objects, 1,
                                     weighing='volume'),
    'CFFOBC' : lambda : ConvexCFF2D('D2Q9', (height, width), objects, 1,
                                     weighing='OBCNone'),
}

#================================================================================
# Main loop

# These are used to monitor time convergence. We save at most 50 snapshots
nTimeSnapshots = min(50, Nt)
timeConvergenceDict = {case : {'Cl' : [], 'Cd' : []} for case in runCases}
saveTimes = np.linspace(0, Nt-1, nTimeSnapshots, dtype=int)

for case in runCases:

    S = cases[case]()

    # Populations at the inflow: equilibrium condition
    fIn = S.fEq(1, np.array([U, 0])) #Inflow f

    # Initialize based on velocity and density
    rho0 = np.ones((1, *S.N, 1))
    U0 = np.ones((1, *S.N, 1))*np.array([U, 0])
    S.initialize(rho0, U0)


    for it in range(Nt): 

        S.collide(tau)

        # Monitor convergence of the lift and drag coefficients in time
        if it in saveTimes:
            # Compute force coefficients
            Cd, Cl = S.Fi2Ftot @ S.F.flatten()/(0.5*1*U**2*c)
            timeConvergenceDict[case]['Cl'].append(Cl)
            timeConvergenceDict[case]['Cd'].append(Cd)

            print(f'{case} : {int(it/Nt*100)}% complete')

        S.stream()       

        #------------------------------------------------------------------------
        #Boundary conditions
        
        #Symmetry bottom
        S.F[2,0] = S.F[4,-1]
        S.F[5,0] = S.F[8,-1]
        S.F[6,0] = S.F[7,-1]

        #Symmetry top
        S.F[4,-1] = S.F[2,0]
        S.F[8,-1] = S.F[5,0]
        S.F[7,-1] = S.F[6,0]

        #Inlet: specified pressure and velocity
        S.F[[1,5,8],:,0,None,:] = fIn[[1,5,8]]

        #Outlet: zero gradient
        S.F[[3,6,7],:,-1,None,:] = S.F[[3,6,7],:,-2,None,:]

    #----------------------------------------------------------------------------
    # Post-processing
        
    # Get velocity field
    Ux = S.getMoments()[1][...,0,None]
    Uy = S.getMoments()[1][...,1,None]

    # Get surface and total forces. These should always be calculated after collision.
    S.collide(tau)
    fsurf = S.Fi2fsurf @ S.F.flatten()
    FTot = S.Fi2Ftot @ S.F.flatten()

    # Convert total force to force coefficients
    L = FTot[1]
    D = FTot[0]
    Cl = L/(1/2*U**2*c)
    Cd = D/(1/2*U**2*c)
    print(f'Cl : {Cl}')
    print(f'Cd : {Cd}')

    # Convert surface forces to surface force coefficients
    p = np.sum(fsurf.reshape(-1, 2)*normalsAirfoil, axis=1)
    Cp = (p - 1/3)/(1/2*U**2) #Pressure coefficient
    tau_w = np.sum(fsurf.reshape(-1, 2)*tangentsAirfoil, axis=1)
    Cf = tau_w/(1/2*U**2) #Skin friction coefficient

    # Plot the convergence of Cl through time
    plt.figure()
    plt.plot(saveTimes, timeConvergenceDict[case]['Cl'])
    plt.xlabel('Timestep')
    plt.ylabel(r'$C_l$')
    plt.ylim(0, timeConvergenceDict[case]['Cl'][-1]*2)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'../timeConvergence/{case}_ClConvergence.pdf')
    plt.clf()

    # Plot the convergence of Cd through time
    plt.figure()
    plt.plot(saveTimes, timeConvergenceDict[case]['Cd'])
    plt.xlabel('Timestep')
    plt.ylabel(r'$C_d$')
    plt.ylim(0, timeConvergenceDict[case]['Cd'][-1]*2)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'../timeConvergence/{case}_CdConvergence.pdf')
    plt.clf()

    # Save post-processed results
    np.save(f'{case}_Cp', Cp)
    np.save(f'{case}_Cf', Cf)
    np.save(f'{case}_Cl', Cl)
    np.save(f'{case}_Cd', Cd)

    # For the CFF case with 16 cells per chord, plot the piecewise polynomials
    # near the leading edge.
    plt.close('all')
    if isinstance(S, ConvexCFF2D) and c == 16:
        extent = (width/4 - 5, width/4 + 5, height/2 - 5, height/2 + 5)
        X1D = np.linspace(extent[0], extent[1], 500)
        Y1D = np.linspace(extent[2], extent[3], 500)
        X, Y = np.meshgrid(X1D, Y1D)
        FInterp = S.interpolatePPs(X, Y)

        plt.figure()
        i = 4
        plt.imshow(FInterp[i], origin='lower', extent=extent, cmap='jet',
                   vmin=0.1, vmax=0.12)
        plt.fill(*airfoilCoordsTrans.T, c='k')
        plt.xlim(*extent[:2])
        plt.ylim(*extent[2:])
        plt.savefig('../../Figures/piecewise_polynomials.pdf')
        plt.show()

        


