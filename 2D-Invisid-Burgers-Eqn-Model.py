from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Set Up
x_max = 2
x_min = 0
y_max = 2
y_min = 0
t_initial = 0
t_end = .5
nx = 101
ny = 101
nt = 101
dx = (x_max - x_min) / (nx-1)
dy = (y_max - y_min) / (ny-1)
dt = (t_end - t_initial) / (nt-1)

#Create arrays
x = np.zeros(nx)
y = np.zeros(nx)
t = np.zeros(nt)
u = np.ones((nt,ny,nx))
u_temp = np.ones((ny,nx))
u_i = np.ones((ny,nx))
v = np.ones((nt,ny,nx))
v_temp = np.ones((ny,nx))
v_i = np.ones((ny,nx))

#Populate dimensional vectors
x = np.linspace(x_min,x_max,num=nx)
y = np.linspace(y_min,y_max,num=ny)
t = np.linspace(t_initial,t_end,num=nt)

#IC's

for i in range(25,75):
    for j in range(25,75):
        u[0][i][j] = 2
        v[0][i][j] = 2

u_i = u[0].copy()
v_i = v[0].copy()

#Iterate through time over spaital (x,y) domain from IC's
for n in range(1,nt):
    print(n)
    u_temp = u_i.copy()
    v_temp = v_i.copy()
    for i in range(1,nx):
        for j in range(1,ny):

            u_i[i][j] = (u_temp[i][j] - (u_temp[i][j] * dt / dx * (u_temp[i][j]
             - u_temp[i - 1][j])) - v_temp[i][j] * dt / dy * (u_temp[i][j]
              - u_temp[i][j-1]))

            v_i[i][j] = (v_temp[i][j] - (u_temp[i][j] * dt / dx * (v_temp[i][j]
             - v_temp[i - 1][j])) - v_temp[i][j] * dt / dy * (v_temp[i][j]
              - v_temp[i][j-1]))

    u[n] = u_i.copy()
    v[n] = v_i.copy()

#Plotting - 3D with z = velocity domain
fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, u[0][:], cmap=cm.viridis)
