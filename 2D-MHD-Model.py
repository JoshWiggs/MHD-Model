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
vis = 0.001
rho = 1
mu = 1
nx = 51
ny = 51
nt = 501
dx = (x_max - x_min) / (nx-1)
dy = (y_max - y_min) / (ny-1)
dt = (t_end - t_initial) / (nt-1)

#Create arrays
x = np.zeros(nx)
y = np.zeros(ny)
t = np.zeros(nt)
u = np.ones((nt,ny,nx))
u_temp = np.ones((ny,nx))
u_i = np.ones((ny,nx))
v = np.ones((nt,ny,nx))
v_temp = np.ones((ny,nx))
v_i = np.ones((ny,nx))
b = np.zeros_like(u_i)

#Populate dimensional vectors
x = np.linspace(x_min,x_max,num=nx)
y = np.linspace(y_min,y_max,num=ny)
t = np.linspace(t_initial,t_end,num=nt)

def magnetic_field_2D(nx,ny,x,y):

    #For A_z = y^2 - x^2

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

    return B_x, B_y, B_mag

B_x, B_y, B_mag = magnetic_field_2D(nx,ny,x,y)

BX,BY = np.meshgrid(B_x,B_y)

#IC's

for i in range(10,25):
    for j in range(10,25):
        u[0][i][j] = 2
        v[0][i][j] = 2

u_i = u[0].copy()
v_i = v[0].copy()

#Iterate through time over spaital (x,y) domain from IC's
for n in range(1,nt):

    if n % 10 == 0:
        print(n)

    u_temp = u_i.copy()
    v_temp = v_i.copy()
    for i in range(1,nx-1):
        for j in range(1,ny-1):

            u_i[i][j] = (u_temp[i][j] + (((vis * dt) / (dx) ** 2) *
            (u_temp[i + 1][j] - 2 * u_temp[i][j] + u_temp[i - 1][j])) +
            (((vis * dt) / (dy) ** 2) * (u_temp[i][j + 1] - 2 * u_temp[i][j] +
            u_temp[i][j - 1])) - (((u_temp[i][j] * dt) / dx) * (u_temp[i][j] -
             u_temp[i - 1][j])) - (((v_temp[i][j] * dt) / dy) * (u_temp[i][j] -
              u_temp[i][j -1])) + (((BX[i][j] * dt) / (rho * mu * dx)) *
              (BX[i][j] - BX[i - 1][j])) + (((BY[i][j] * dt) / (rho * mu * dy))
              * (BX[i][j] - BX[i][j - 1])) + ((dt / (2 * rho * mu * dx)) *
              (B_mag[i][j] ** 2 - B_mag[i - 1][j] ** 2)))

            v_i[i][j] = (v_temp[i][j] + (((vis * dt) / (dx) ** 2) *
             (v_temp[i + 1][j] - 2 * v_temp[i][j] + v_temp[i - 1][j])) +
             (((vis * dt) / (dy) ** 2) * (v_temp[i][j + 1] - 2 * v_temp[i][j] +
             v_temp[i][j - 1])) - (((u_temp[i][j] * dt) / dx) * (v_temp[i][j] -
             v_temp[i - 1][j])) - (((v_temp[i][j] * dt) / dy) * (v_temp[i][j] -
              v_temp[i][j -1])) + (((BX[i][j] * dt) / (rho * mu * dx)) *
              (BY[i][j] - BY[i - 1][j])) + (((BY[i][j] * dt) / (rho * mu * dy))
              * (BY[i][j] - BY[i][j - 1])) + ((dt / (2 * rho * mu * dy)) *
              (B_mag[i][j] ** 2 - B_mag[i][j - 1] ** 2)))

    u[n] = u_i.copy()
    v[n] = v_i.copy()

#Meshgrid
X, Y = np.meshgrid(x, y)

#Plotting - 3D with z = velocity domain
def time_plot(r,t):
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, r[t][:], cmap=cm.viridis)
    plt.show()
