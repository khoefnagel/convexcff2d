"""
Run different boundary treatments for the case of pure advection in a 2D inclined channel. 
The following case parameters should be defined in the file ../caseDef.json:
- N : Number of cell sizes between the two walls
- width : Number of fluid nodes in the x direction
- height : Number of fluid nodes in the y direction
- thetaDeg : Angle of the channel with the x axis in degrees
- Nt : Number of timesteps to run

The population fields are saved in files F_<case>.npy

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

#Load variables defined in the caseDef json file into local namespace
with open('../caseDef.json') as f:
    locals().update(json.load(f))

#Set up the geometry
theta = thetaDeg*np.pi/180

b = N//2 # Vertical solid portion on bottom left
a = height - b - N/np.cos(theta) # Vertical solid portion on top left

# Locations where the walls intersect the domain boundaries
x1, x2, x3, x4 = 0, a/np.tan(theta), width-b/np.tan(theta), width
y1, y2, y3, y4 = 0, b, height-a, height

# Define the three polygons that make up the geometry
lowerTriangle = np.array([[x3, y1], [x4, y1], [x4, y2], [x3, y1]])
center = np.array([[x1, y1], [x2, y1], [x4, y3], [x4, y4],
                   [x3, y4], [x1, y2], [x1, y1]],)
upperTriangle = np.array([[x1, y3], [x2, y4], [x1, y4], [x1, y3]])

geom = [lowerTriangle, center, upperTriangle]


cases = {
    'Chen' : lambda : ConvexChen1998('D2Q9', (height, width), geom),
    'CFFNone' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 1,
                                      weighing=None),    
    'CFFVol' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 1,
                                     weighing='volume'),
    'CFFOBC' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 1,
                                     weighing='OBCNone'),
    'CFFConst' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 0),
    'CFFNoneXY' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 1,
                                        weighing=None, xy=True), 
    'CFFVolXY' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 1,
                                       weighing='volume', xy=True),
    'CFFOBCXY' : lambda : ConvexCFF2D('D2Q9', (height, width), geom, 1,
                                       weighing='OBCNone', xy=True),
}



#================================================================================
#Exact solution function

def getFExact(X, Y, theta, N, a, b, height):
    xWidth = N/np.sin(theta)

    bottomRightInds = np.where(np.tan(theta)*X > Y)
    topLeftInds = np.where(np.tan(theta)*X <= Y)
    
    Fx = np.zeros(X.shape)
    Fx[bottomRightInds] = 0.5 - 0.5*np.cos(2*np.pi/xWidth*(X[bottomRightInds] -\
                                           (Y[bottomRightInds] + a)/np.tan(theta)))
    Fx[topLeftInds] = 0.5 - 0.5*np.cos(2*np.pi/xWidth*(X[topLeftInds] - (Y[topLeftInds] - b - N/np.cos(theta))/np.tan(theta)))
    
    Fy = 0.5 - 0.5*np.cos(2*np.pi/height*Y)
    return Fx*Fy

#================================================================================
#Function to get the x and y coordinate at t=0 for a point (xp, yp) at t=Nt

def getXYt0(xp, yp, theta, N, a, b, Nt, T13, T24, T57, T68):

    #Modulo of Nt with each fundamental period
    NtMod_i13 = Nt%(T13)
    NtMod_i24 = Nt%(T24)
    NtMod_i57 = Nt%(T57)
    NtMod_i68 = Nt%(T68)

    XYt0 = np.ones((9, 2))*np.nan
    
    #i = 0
    XYt0[0] = np.array([xp, yp])

    #========================================================================
    #i = 1,3
    if np.tan(theta)*xp > yp: #Bottom right
        xmax = 1/np.tan(theta)*(yp + a + N/np.cos(theta))
    else: #Top left
        xmax = 1/np.tan(theta)*(yp - b)
    xmin = xmax - T13/2
    
    #i = 1
    if xp - xmin < NtMod_i13 < T13/2 + xp - xmin:
        x = xmin + (NtMod_i13 - (xp - xmin))
    else:
        x = xmin + ((xp - xmin) - NtMod_i13)%(T13/2)
    XYt0[1] = np.array([x, yp])

    #i = 3
    if xmax - xp < NtMod_i13 < T13/2 + xmax - xp:
        x = xmax - (NtMod_i13 - (xmax - xp))
    else:
        x = xmin + ((xp - xmin) + NtMod_i13)%(T13/2)
    XYt0[3] = np.array([x, yp])

    #========================================================================
    #i = 2,4
    if np.tan(theta)*xp > yp: #Bottom right
        ymax = np.tan(theta)*xp - a
    else: #Top left
        ymax = np.tan(theta)*xp + b + N/np.cos(theta)
    ymin = ymax - T24/2
    
    #i = 2
    if yp - ymin < NtMod_i24 < T24/2 + yp - ymin:
        y = ymin + (NtMod_i24 - (yp - ymin))
    else:
        y = ymin + ((yp - ymin) - NtMod_i24)%(T24/2)
    XYt0[2] = np.array([xp, y])

    #i = 4
    if ymax - yp < NtMod_i24 < T24/2 + ymax - yp:
        y = ymax - (NtMod_i24 - (ymax - yp))
    else:
        y = ymin + ((yp - ymin) + NtMod_i24)%(T24/2)
    XYt0[4] = np.array([xp, y])

    #========================================================================
    #i = 5,7
    if np.tan(theta)*xp > yp: #Bottom right
        xmax = 1/(1 - np.tan(theta))*(xp - yp - a)
    else: #Top left
        xmax = 1/(1 - np.tan(theta))*(xp - yp + b + N/np.cos(theta))
    xmin = xmax - T57/2
    ymax = xmax + yp - xp
    ymin = xmin + yp - xp

    #i = 5
    if xp - xmin < NtMod_i57 < T57/2 + xp - xmin:
        x = xmin + (NtMod_i57 - (xp - xmin))
        y = ymin + (NtMod_i57 - (yp - ymin))
    else:
        x = xmin + ((xp - xmin) - NtMod_i57)%(T57/2)
        y = x + yp - xp
    XYt0[5] = np.array([x, y])

    #i = 7
    if xmax - xp < NtMod_i57 < T57/2 + xmax - xp:
        x = xmax - (NtMod_i57 - (xmax - xp))
        y = ymax - (NtMod_i57 - (ymax - yp))
    else:
        x = xmin + ((xp - xmin) + NtMod_i57)%(T57/2)
        y = x + yp - xp        
    XYt0[7] = np.array([x, y])

    #========================================================================
    #i = 6,8
    if np.tan(theta)*xp > yp: #Bottom right
        xmin = 1/(1 + np.tan(theta))*(xp + yp + a)
    else: #Top left
        xmin = 1/(1 + np.tan(theta))*(xp + yp - b - N/np.cos(theta))
    xmax = xmin + T68/2
    ymax = -xmin + yp + xp
    ymin = -xmax + yp + xp

    #i = 6
    if xmax - xp < NtMod_i68 < T68/2 + xmax - xp:
        x = xmax - (NtMod_i68 - (xmax - xp))
        y = ymin + (NtMod_i68 - (yp - ymin))
    else:
        x = xmin + ((xp - xmin) + NtMod_i68)%(T68/2)
        y = -x + yp + xp
    XYt0[6] = np.array([x, y])

    #i = 8
    if xp - xmin < NtMod_i68 < T68/2 + xp - xmin:
        x = xmin + (NtMod_i68 - (xp - xmin))
        y = ymax - (NtMod_i68 - (ymax - yp))
    else:
        x = xmin + ((xp - xmin) - NtMod_i68)%(T68/2)
        y = -x + yp + xp       
    XYt0[8] = np.array([x, y])

    return XYt0

#================================================================================
#Function to get the exact population vector at Nt

def getFExactNt(Nt, theta, N, a, b, height, X, Y):

    
    #Fundamental periods for each opposite population pair
    T13 = 2*N/np.sin(theta)
    T24 = 2*N/np.cos(theta)
    T57 = np.sqrt(2)*N/np.sin(np.pi/4-theta)
    T68 = np.sqrt(2)*N/np.cos(np.pi/4-theta)

    #Initialize
    FExact = np.ones((9, height, width, 1))*np.nan

    #Loop over each point index
    for xi, yi in zip(*[inds.flatten() for inds in\
                    np.meshgrid(np.arange(width), np.arange(height))]):
        
        #Get the point coordinates
        xp, yp = X[yi, xi], Y[yi, xi]

        #Trace the point back in time and get corresponding coordinates 
        #(x, y) at t=0
        x, y = getXYt0(xp, yp, theta, N, a, b, Nt, T13, T24, T57, T68).T
        
        #Calculate FExact at t=0 for these coordinates. Since streaming is pure
        #advection at t=Nt, the value is the same at (xp, yp).
        FExact[:, yi, xi, 0] = getFExact(x, y, theta, N, a, b, height)
    
    return FExact


#================================================================================
#Main loop

for case in runCases:

    S = cases[case]()

    #Cell coorinates correspond to the centroids
    X, Y = S.centroidsGlob[0,:,:,0], S.centroidsGlob[0,:,:,1]
   
    #Initialize F
    S.F[:,:,:,0] = getFExact(X, Y, theta, N, a, b, height)


    for it in range(Nt): #Time iteration

        #Only streaming
        S.stream()

    np.save(f'{case}_F', S.F)

#================================================================================
#Get the exact solution at t=Nt and save

FExact = getFExactNt(Nt, theta, N, a, b, height, X, Y)
np.save(f'FExact', FExact)

np.save('cellVolumes', S.cellVolumes)
