"""
Script to load in the airfoil coordinates and convert them to a geo file
which is interpretable by gmsh. The file NACA0012.geo is created.

Requires: numpy
"""

import numpy as np

# Load in the case variables
with open('./caseDef', 'r') as f:
    lines = f.readlines()
for line in lines:
    var, val = line.split(';')[0].split(' ')
    locals().update({var : float(val)})


alphaRad = np.deg2rad(alphaDeg)

# Load airfoil coordinates
data = np.loadtxt('./NACA0012.dat', skiprows=1)

# Translate
data[:,0] = data[:,0] - 0.25 

#Scale
data = data*c

data2 = np.array(data) #Deep copy

#Rotate
data2[:,0] = data[:,0]*np.cos(alphaRad) + data[:,1]*np.sin(alphaRad)
data2[:,1] = -data[:,0]*np.sin(alphaRad) + data[:,1]*np.cos(alphaRad)



top = data2[:len(data)//2]
bottom = data2[len(data)//2:]

# Output file
file = open('NACA0012.geo', 'w')

# Write top vertices
for i in range(len(top)):

    point = 'Point(' + str(i+1) + ') = '
    coords = '{' + str(top[i][0]) + ', ' + str(top[i][1]) + ', ' + '0};' + '\n'

    file.write(point+coords)

# Write bottom vertices
for i in range(len(bottom)):

    point = 'Point(' + str(i+1+len(top)) + ') = '
    coords = '{' + str(bottom[i][0]) + ', ' + str(bottom[i][1]) + ', ' + '0};' + '\n'

    file.write(point+coords)

# Create the lines between vertices
Line = 'Line(1) = {'

for i in range(len(top) + len(bottom)):
    if not(i == len(top) + len(bottom) - 1):
        Line = Line + str(i+1) + ', '
    else:
        Line = Line + str(i+1) + ', 1};'

file.write(Line)
file.close()
    
