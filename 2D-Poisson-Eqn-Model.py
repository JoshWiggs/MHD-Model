from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Set Up
x_max = 2
x_min = 0
y_max = 1
y_min = 0
nx = 51
ny = 51
nt = 100
dx = (x_max - x_min) / (nx-1)
dy = (y_max - y_min) / (ny-1)

#Create arrays
x = np.zeros(nx)
y = np.zeros(nx)
p = np.zeros((nx,ny))
p_temp = np.zeros((nx,ny))
p_initial = np.zeros((nx,ny))
inp = np.zeros_like(p) #driver

#Populate dimensional vectors
x = np.linspace(x_min,x_max,num=nx)
y = np.linspace(y_min,y_max,num=ny)

# Source
inp[int(nx / 4), int(ny / 4)]  = 100
inp[int(3 * nx / 4), int(3 * ny / 4)] = -100

for t in range(nt):
    
    if t % 10 == 0:
        print(t)
        
    for i in range(1,nx-1):
        for j in range(1,ny-1):

            p_temp = p.copy()

            p[i][j] = ((p_temp[i + 1][j] + p_temp[i - 1][j]) * (dy ** 2) +
            (p_temp[i][j + 1] + p_temp[i][j - 1]) * (dx ** 2) - (inp[i][j] *
            (dx ** 2) * (dy ** 2))) / (2 * ((dx ** 2) + (dy ** 2)))

            p[0, :] = 0
            p[ny-1, :] = 0
            p[:, 0] = 0
            p[:, nx-1] = 0

#Meshgrid
X, Y = np.meshgrid(x, y)

#Plotting - 3D with z = velocity domain
def sur_plot(r):
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, r, cmap=cm.viridis)
    plt.show()
