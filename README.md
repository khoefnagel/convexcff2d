# ConvexCFF2D

This repository contains all code and data used in the publication: "A second-order volumetric boundary treatment for the lattice Boltzmann method".

The source code of the Continuum Field Formulation (CFF) boundary treatment is contained in the src directory.
The testcases (both full lattice Boltzmann and pure advection) can be found in the testcases directory.
The eigenvalue analysis of the streaming operation can be found in the eigenvalues directory.

## Requirements
* Python version 3.10 or higher
* Python libraries:
	* numpy
	* scipy
	* matplotlib
	* sympy
* (OpenFOAM v7) => Only to generate airfoil reference data
* (Paraview)

## Figures
The scripts that were used to generate the results figures can be found as follows:
* Fig. 10: ./eigenvalues/2D_inclined_channel/cases/theta30/N008/plotEigenvalues.py
* Fig. 11a: ./eigenvalues/2D_inclined_channel/cases/theta30/N008/6-8/plotEigenMode.py
* Fig. 11b: ./eigenvalues/2D_inclined_channel/cases/theta30/N008/1-3/plotEigenMode.py
* Fig. 11c: ./eigenvalues/2D_inclined_channel/cases/theta30/N008/5-7/plotEigenMode.py
* Fig. 12: ./eigenvalues/2D_inclined_channel/cases/theta30/N008/plotEigenvalues.py
* Fig. 14: ./testcases/full_LBM/quasi_1D_channel/cases/plotAllCases.py
* Fig. 16: ./testcases/advection_only/1D/cases/plotAllCases.py
* Fig. 17: ./testcases/full_LBM/inclined_channel_2D/cases/plotAllCases.py
* Fig. 18: ./testcases/advection_only/2D_inclined_channel/cases/theta30/plotThetaCase.py
* Fig. 20a-d: ./testcases/full_LBM/NACA0012_2D/cases/alpha005/compareClCd.py
* Fig. 20e-f: ./testcases/full_LBM/NACA0012_2D/cases/alpha005/compareCpCf.py
* Fig. 21: ./testcases/full_LBM/NACA0012_2D/cases/alpha005/c016/CFF/plotPressure.py
* Fig. 22: ./testcases/full_LBM/NACA0012_2D/cases/alpha005/c016/CFF/runCFF.py


## Tables
The scripts that were used to generate the results tables can be found as follows:
* Tab. II: ./testcases/full_LBM/quasi_1D_channel/cases/plotAllCases.py
* Tab. III: ./testcases/full_LBM/inclined_channel_2D/cases/plotAllCases.py
* Tab. IV: ./testcases/full_LBM/NACA0012_2D/cases/alpha005/c064/CFF/checkComputationalCost.py


## Claims
Claims regarding our results are backed up as follows:
* Boundary weighted, where only pgram-overlapped cells are used in the projection, makes the method unstable: ./eigenvalues/1D/cases/checkStability.py
* Including an xy component in the projection makes the method unstable: ./eigenvalues/2D_inclined_channel/cases/checkStability.py
* For the 1D pure advection case, we find 3rd order convergence in L1: ./testcases/advection_only/1D/cases/plotAllCases.py
* After an initialization effect, a third-order convergence is observed in the cells adjacent to the cut cells, which propagates to the bulk: ./derivations/advection_only_convergence_rate/analytical_error_1D.py


