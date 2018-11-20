from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Set Up
x_max = 2
x_min = 0
y_max = 2
y_min = 0
nx = 101
ny = 101
dx = (x_max - x_min) / (nx-1)
dy = (y_max - y_min) / (ny-1)

#Create arrays
x = np.zeros(nx)
y = np.zeros(ny)

#Populate dimensional vectors
x = np.linspace(x_min,x_max,num=nx)
y = np.linspace(y_min,y_max,num=ny)


def magnetic_field_2D(nx,ny,x,y):
        
    B_x = np.zeros(nx)
    B_y = np.zeros(ny)
    B_mag = np.zeros((nx,ny))

    for i in range(0,nx):

        B_x[i] = (2 * y[i])

    for j in range(0,ny):

        B_y[j] = (2 * x[j])

    for i in range(0,nx):
        for j in range(0,ny):

            B_mag[i][j] = np.sqrt(B_x[i]**2 + B_x[j]**2)

    X,Y = np.meshgrid(x,y)
    plt.contourf(X,Y,B_mag)
    plt.show()
    


magnetic_field_2D(nx,ny,x,y)






