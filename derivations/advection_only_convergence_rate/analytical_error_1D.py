'''
Script to analytically derive the convergence rate of volume-weighted CFF
for the advection-only 1D test case. This script uses sympy to perform
symbolic calculations.
Because the subzones are different for pfluid < 0.5 and pfluid > 0.5,
we explicitly distinguish these two cases where necessary.
The exact solution used here is a sine function.

The analytical expression for the error as a function of cell size h is
written to the file errorConvergence.txt

Requires: sympy, numpy, sys
'''

import sympy as sym
import numpy as np
import sys


#===================================================================================
#Parameters

#Fluid fraction of the cut cell
pfluid = 1/3

# Phase shift for the exact solution
phase_shift = 0.3

#Number of timesteps. Since the equations cascade, don't put this too high.
Nt = 8

#Weight of the cut cell in the least squares (normalized such that the weight of the full cell is 1)
w0 = pfluid/(pfluid+1)

#===================================================================================
#Symbols

# Incoming populations in cut cell, adjacent cell (x1) and next cell (x2)
f_2_x0, f_2_x1, f_2_x2 = sym.symbols('f_2_x0 f_2_x1 f_2_x2')

# Outgoing populations in cut cell
f_1_x0 = sym.Symbol('f_1_x0')

# Grid spacing (referred to as Delta in the paper)
h = sym.Symbol('h')

# Spatial coordinate
x = sym.Symbol('x')

#===================================================================================
#Relevant coordinates

#Cell centers
x0 = h - h*pfluid/2
x1 = 1.5*h
x2 = 2.5*h

#Subzone boundaries
xb = h*(1 - pfluid) # Facet location
xe = xb + h # End of the pgram
xcrefl = 2*xb # Reflected cell border

#===================================================================================
#Piecewise polynomial function in cell 0. Origin at centroid of cell 0.

a_x0 = (f_2_x1 - f_2_x0)/(x1 - x0)
ftilde_2_x0 = a_x0*(x - x0) + f_2_x0

#===================================================================================
#Piecewise polynomial function in cell 1. Since there are two neighbours,
# we use a weighted least-squares approach to find the slope.

#Weighted least-squares
X = np.array([[(x0 - x1)], #x0 - x1
               [x2 - x1]]) #LHS matrix
W = np.array([[w0, 0], [0, 1-w0]]) #w0 + w2 === 1 => w2 = 1-w0
y = np.array([[f_2_x0 - f_2_x1],
              [f_2_x2 - f_2_x1]]) #RHS vector

#Slope based on weighted-least squares
a_x1 = (1/(X.T @ W @ X) @ X.T @ W @ y)[0,0]

#Piecewise polynomial function in cell 1. Origin at centroid of cell 1
ftilde_2_x1 = a_x1*(x - x1) + f_2_x1


#===================================================================================
#Integrals over subzones of the piecewise polynomials, for pfluid < 0.5

#Integral of ftilde_2_x0 over x0
int_x0_spf = sym.integrate(ftilde_2_x0, (x, xb, h))

#Integral of ftilde_2_x1 over x1 from the left border to Crefl
int_x1_spf_1 = sym.integrate(ftilde_2_x1, (x, h, xcrefl))

#Integral of ftilde_2_x1 over x1 from Crefl to the end of the pgram
int_x1_spf_2 = sym.integrate(ftilde_2_x1, (x, xcrefl, xe))

#Integral of ftilde_2_x1 over x1 from the end of the pgram to the right cell border
int_x1_spf_3 = sym.integrate(ftilde_2_x1, (x, xe, 2*h))

#===================================================================================
#Integrals over subzones of the piecewise polynomials, for pfluid > 0.5

#Integral of ftilde_2_x0 over x0 from the boundary to Crefl
int_x0_lpf_1 = sym.integrate(ftilde_2_x0, (x, xb, xcrefl))

#Integral of ftilde_2_x0 over x0 from Crefl to right cell border
int_x0_lpf_2 = sym.integrate(ftilde_2_x0, (x, xcrefl, h))

#Integral of ftilde_2_x1 over x1 from the left border to the end of the pgram
int_x1_lpf_1 = sym.integrate(ftilde_2_x1, (x, h, xe))

#Integral of ftilde_2_x1 over x1 from the end of the pgram to the right cell border
int_x1_lpf_2 = sym.integrate(ftilde_2_x1, (x, xe, 2*h))


#===================================================================================
#Exact solution

#Sine function
f_2_exact = [0.5 - 0.5*sym.cos(2*sym.pi*(x-h*(1-pfluid) + phase_shift))]
f_1_exact = [0.5 - 0.5*sym.cos(2*sym.pi*(x-h*(1-pfluid) - phase_shift))]

# Alternative: third order function 
#f_2_exact = [(x - xb + np.sqrt(0.5))**3 + np.pi*(x - xb + np.sqrt(0.5))**2 + np.exp(1)*(x - xb + np.sqrt(0.5))]
#f_1_exact = [(xb - x + np.sqrt(0.5))**3 + np.pi*(xb - x + np.sqrt(0.5))**2 + np.exp(1)*(xb - x + np.sqrt(0.5))]

# Alternative: second order function 
#f_2_exact = [(x - xb + np.sqrt(0.5))**2]
#f_1_exact = [(xb - x + np.sqrt(0.5))**2]

# Alternative: first order function 
#f_2_exact_pre = x + 0.5 - h*(1-pfluid)
#f_1_exact_pre = -x + 0.5 + h*(1-pfluid)

for t in range(1, Nt):
    f_2_exact.append(f_2_exact[0].subs(x, x+t*h))
    f_1_exact.append(f_1_exact[0].subs(x, x-t*h))

# Exact solution integrated over the cells
fbar_2_x0_exact = [sym.integrate(f_2_exact[ti], (x, xb, h))/(pfluid*h)\
                   for ti in range(Nt)]
fbar_1_x0_exact = [sym.integrate(f_1_exact[ti], (x, xb, h))/(pfluid*h)\
                   for ti in range(Nt)]
fbar_1_x1_exact = [sym.integrate(f_1_exact[ti], (x, h, 2*h))/h\
                   for ti in range(Nt)]


#===================================================================================
#Timestepping

#Initialize with cell values at t=0
fbar_2_x0 = [sym.integrate(f_2_exact[0], (x, xb, h))/(pfluid*h)]
fbar_1_x0 = [sym.integrate(f_1_exact[0], (x, xb, h))/(pfluid*h)]
fbar_1_x1 = [sym.integrate(f_1_exact[0], (x, h, 2*h))/h]

#Create copies for pfluid < 0.5 and pfluid > 0.5
fbar_2_x0_spf, fbar_2_x0_lpf = fbar_2_x0.copy(), fbar_2_x0.copy()
fbar_1_x0_spf, fbar_1_x0_lpf = fbar_1_x0.copy(), fbar_1_x0.copy()
fbar_1_x1_spf, fbar_1_x1_lpf = fbar_1_x1.copy(), fbar_1_x1.copy()

for it in range(Nt):

    fbar_2_x1 = sym.integrate(f_2_exact[it], (x, h, 2*h))/h
    fbar_2_x2 = sym.integrate(f_2_exact[it], (x, 2*h, 3*h))/h
    
    f_exact_dict_spf  = {f_2_x0 : fbar_2_x0_spf[it], f_2_x1 : fbar_2_x1,
                         f_2_x2 : fbar_2_x2, f_1_x0 : fbar_1_x0_spf[it]}
    f_exact_dict_lpf  = {f_2_x0 : fbar_2_x0_lpf[it], f_2_x1 : fbar_2_x1,
                         f_2_x2 : fbar_2_x2, f_1_x0 : fbar_1_x0_lpf[it]}

    #Geometric mapping of subzone integrals, pfluid < 0.5
    fbar_2_x0_spf.append((int_x1_spf_3/(h*pfluid)).subs(f_exact_dict_spf))
    fbar_1_x0_spf.append((int_x1_spf_2/(h*pfluid)).subs(f_exact_dict_spf))
    fbar_1_x1_spf.append(((int_x1_spf_1 + int_x0_spf + f_1_x0*h*pfluid)/h)\
                         .subs(f_exact_dict_spf))

    #Geometric mapping of subzone integrals, pfluid > 0.5
    fbar_2_x0_lpf.append((int_x1_lpf_2/(h*pfluid)).subs(f_exact_dict_lpf))
    fbar_1_x0_lpf.append(((int_x1_lpf_1 + int_x0_lpf_2)/(h*pfluid))\
                          .subs(f_exact_dict_lpf))
    fbar_1_x1_lpf.append(((int_x0_lpf_1 + f_1_x0*h*pfluid)/h).subs(f_exact_dict_lpf))

#===================================================================================
#Calculate errors

#pfluid < 0.5
fbar_2_x0_err_spf = []
fbar_1_x0_err_spf = []
fbar_1_x1_err_spf = []

#pfluid > 0.5
fbar_2_x0_err_lpf = []
fbar_1_x0_err_lpf = []
fbar_1_x1_err_lpf = []

#Timestepping
for it in range(Nt):

    #pfluid < 0.5
    fbar_2_x0_err_spf.append(sym.Abs(fbar_2_x0_exact[it] - fbar_2_x0_spf[it]))
    fbar_1_x0_err_spf.append(sym.Abs(fbar_1_x0_exact[it] - fbar_1_x0_spf[it]))
    fbar_1_x1_err_spf.append(sym.Abs(fbar_1_x1_exact[it] - fbar_1_x1_spf[it]))

    #pfluid > 0.5
    fbar_2_x0_err_lpf.append(sym.Abs(fbar_2_x0_exact[it] - fbar_2_x0_lpf[it]))
    fbar_1_x0_err_lpf.append(sym.Abs(fbar_1_x0_exact[it] - fbar_1_x0_lpf[it]))
    fbar_1_x1_err_lpf.append(sym.Abs(fbar_1_x1_exact[it] - fbar_1_x1_lpf[it]))

    with open('errorConvergence.txt', 'a') as f:

        # Pipe printed output to file
        sys.stdout = f
        
        print('time = ', it)
        print('-------')
        if pfluid < 0.5:
            print('Error of incoming populations in the cut cell:')
            print(sym.nsimplify(fbar_2_x0_err_spf[it].series(h, 0, 4), tolerance=1e-13).evalf(6))
            print('\nError of outgoing populations in the cut cell:')
            print(sym.nsimplify(fbar_1_x0_err_spf[it].series(h, 0, 4), tolerance=1e-13).evalf(6))
            print('\nError of outgoing populations in adjacent cell (this propagates to the bulk):')
            print(sym.nsimplify(fbar_1_x1_err_spf[it].series(h, 0, 4), tolerance=1e-13).evalf(6))
        else:
            print('Error of incoming populations in the cut cell:')
            print(sym.nsimplify(fbar_2_x0_err_lpf[it].series(h, 0, 4), tolerance=1e-13).evalf(6))
            print('\nError of outgoing populations in the cut cell:')
            print(sym.nsimplify(fbar_1_x0_err_lpf[it].series(h, 0, 4), tolerance=1e-13).evalf(6))
            print('\nError of outgoing populations in adjacent cell (this propagates to the bulk):')
            print(sym.nsimplify(fbar_1_x1_err_lpf[it].series(h, 0, 4), tolerance=1e-13).evalf(6))
        print('\n')

    sys.stdout = sys.__stdout__

print('Analytical error expressions written to errorConvergence.txt')
