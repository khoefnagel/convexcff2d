'''
Simple script to convert the VTK data of the airfoil to a csv file with data
over the airfoil surface.

Requires: paraview, glob
'''

from paraview.simple import *
import glob

files = glob.glob('./VTK/Airfoil/Airfoil_*.vtk')

# Figure out the last time available
times = [int(file.split('_')[-1].split('.vtk')[0]) for file in files]
lastTime = max(times)

# Read airfoil data
airfoil_vtk = LegacyVTKReader(FileNames=['./VTK/Airfoil/Airfoil_{}.vtk'\
                                         .format(lastTime)])

SaveData('./Surface_data.csv', proxy=airfoil_vtk, Precision=15)


