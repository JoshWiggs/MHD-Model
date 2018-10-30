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
dx = (abs(x_max) + abs(x_min)) / (nx-1)
dy = (abs(y_max) + abs(y_min)) / (ny-1)

#Create arrays
x = np.zeros(nx)
x_temp = np.zeros(nx)
y = np.zeros(nx)
y_temp = np.zeros(nx)
u = np.zeros((ny,nx))
u_temp = np.zeros((ny,nx))

#Populate arrays
x = np.linspace(x_min,x_max,num=nx)
y = np.linspace(y_min,y_max,num=ny)

for i in range(0,nx):
    u[i][0] = 1

for j in range(0,ny):
    u[0][j] = 2

#Iterate
for i in range(1,nx):
    for j in range(1,ny):
        u_temp = u.copy()

        u[i][j] = ((dx * u_temp[i - 1][j]) + (dy * u_temp[i][j - 1])) / (dx + dy)

#Plotting
fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)
