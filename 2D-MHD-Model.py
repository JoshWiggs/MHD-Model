from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from decimal import *

def magnetic_field_2D(nx,ny,x,y):

    B_x = np.zeros(nx)
    B_y = np.zeros(ny)
    B_mag = np.zeros((nx,ny))

    for i in range(0,nx):

        B_x[i] = 0

    for j in range(0,ny):

        B_y[j] = 0

    for i in range(0,nx):
        for j in range(0,ny):

            B_mag[i][j] = np.sqrt(B_x[i]**2 + B_y[j]**2)

    return B_x, B_y, B_mag

def pressure_driving_force_2D(rho,nx,ny,p_nt,dt,dx,dy,u_i,v_i):

    b = np.zeros((nx,ny))
    p_i = np.zeros((nx,ny))

    for i in range(1,nx-1):
        for j in range(1,ny-1):

            b[i][j] = ((rho / dt) * ((u_i[i + 1][j] - u_i[i - 1][j]) / (2 * dx)
             + (v_i[i][j + 1] - v_i[i][j - 1]) / (2 * dy)))

    for n in range(p_nt):
        for i in range(1,nx-1):
            for j in range(1,ny-1):

                p_i[i][j] = (((p_i[i + 1][j] - p_i[i - 1][j]) * dy ** 2  + (p_i[i][j + 1]
                 - p_i[i][j - 1]) * dx ** 2 - (b[i][j] * dx ** 2 * dy ** 2)) /
                 (2 * (dx ** 2 + dy ** 2)))

    p_i[0, :] = 0
    p_i[nx-1, :] = 0
    p_i[:, 0] = 0
    p_i[:, ny-1] = 0

    return b, p_i

def convection_2D(nx,ny,dt,dx,dy,u_i,v_i):

    u_conv = np.zeros((nx,ny))
    v_conv = np.zeros((nx,ny))

    for i in range(1,nx-1):
        for j in range(1,ny-1):

            u_conv[i][j] = - (((u_i[i][j] * dt) / dx) * (u_i[i][j] -
            u_i[i - 1][j])) - (((v_i[i][j] * dt) / dy) * (u_i[i][j] -
            u_i[i][j-1]))

            v_conv[i][j] = - (((u_i[i][j] * dt) / dx) * (v_i[i][j] -
            v_i[i - 1][j])) - (((v_i[i][j] * dt) / dy) * (v_i[i][j] -
            v_i[i][j-1]))

    return u_conv, v_conv

def diffusion_2D(vis,nx,ny,dt,dx,dy,u_i,v_i):

    u_diff = np.zeros((nx,ny))
    v_diff = np.zeros((nx,ny))

    for i in range(1,nx-1):
        for j in range(1,ny-1):

            u_diff[i][j] = (((vis * dt) / dx ** 2) * (u_i[i + 1][j] -
            (2 * u_i[i][j]) + u_i[i - 1][j])) +  (((vis * dt) / dy ** 2) *
            (u_i[i][j + 1] - (2 * u_i[i][j]) + u_i[i][j-1]))

            v_diff[i][j] = (((vis * dt) / dx ** 2) * (v_i[i + 1][j] - (2 *
            v_i[i][j]) + v_i[i - 1][j])) + (((vis * dt) / dy ** 2) *
            (v_i[i][j + 1] - (2 * v_i[i][j]) + v_i[i][j-1]))

    return u_diff ,v_diff

def magnetic_2D(rho,mu,nx,ny,dt,dx,dy,BX,BY,B_mag):

    u_mag = np.zeros((nx,ny))
    v_mag = np.zeros((nx,ny))

    for i in range(1,nx-1):
        for j in range(1,ny-1):

            u_mag[i][j] = (((BX[i][j] * dt) / (rho * mu * dx)) * (BX[i][j] -
            BX[i - 1][j])) + (((BY[i][j] * dt) / (rho * mu * dy)) * (BX[i][j] -
            BX[i][j - 1])) + ((dt / (2 * rho * mu * dx)) * ((B_mag[i][j]) ** 2 -
            (B_mag[i - 1][j]) ** 2))

            v_mag[i][j] = (((BX[i][j] * dt) / (rho * mu * dx)) * (BY[i][j] -
            BY[i - 1][j])) + (((BY[i][j] * dt) / (rho * mu * dy)) * (BY[i][j] -
            BY[i][j - 1])) + ((dt / (2 * rho * mu * dy)) * ((B_mag[i][j]) ** 2 -
            (B_mag[i][j - 1]) ** 2))

    return u_mag, v_mag

#from poisson_solver import pressure_driving_force_2D

#Set Up
x_max = 2
x_min = -2
y_max = 2
y_min = -2
t_initial = 0
t_end = 1
vis = 0.1 #0.001
rho = 10 #1
mu = 1 #1
nx = 101
ny = 101
nt = 1501
p_nt = 101 #peudo-time for pressure
dx = (x_max - x_min) / (nx-1)
dy = (y_max - y_min) / (ny-1)
dt = (t_end - t_initial) / (nt-1)

#Control size of perturbation down interface
amp = 2
freq = 2
#Control actual number of time steps simulation runs over
sim_t = 501

#Create arrays
x = np.zeros(nx)
y = np.zeros(ny)
t = np.zeros(nt)
u = np.ones((sim_t,nx,ny))
u_temp = np.ones((nx,ny))
u_i = np.ones((nx,ny))
u_new = np.ones((nx,ny))
v = np.zeros((sim_t,nx,ny))
v_temp = np.zeros((nx,ny))
v_i = np.zeros((nx,ny))
v_new = np.zeros((nx,ny))
U = np.zeros((sim_t,nx,ny))
p = np.zeros((sim_t,nx,ny))



#Populate dimensional vectors
x = np.linspace(x_min,x_max,num=nx)
y = np.linspace(y_min,y_max,num=ny)
t = np.linspace(t_initial,t_end,num=nt)

B_x, B_y, B_mag = magnetic_field_2D(nx,ny,x,y)

BX,BY = np.meshgrid(B_x,B_y)

#IC's
"""
for i in range(10,50):
    for j in range(10,50):
        u[0][i][j] = 2
        v[0][i][j] = 2


"""
for i in range(0,51):
     u[0][:][i] = -0.25
     u[0][:][50 + i] = 0.25

     xi = np.zeros(nx)
     for i in range(0,nx):
         xi[i] = amp * (np.sin((np.pi * (freq * x[i]))))

     xi[0:25] = 0
     xi[76:101] = 0

     v[0][:][49] = xi
     v[0][:][50] = xi
     v[0][:][51] = xi

u_i = u[0].copy()
v_i = v[0].copy()

#Iterate through time over spaital (x,y) domain from IC's
for n in range(1,sim_t):

    if n % 10 == 0:
        print(n)
    """
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

            #u_i[i][j] = Decimal(u_i[i][j])
            #v_i[i][j] = Decimal(v_i[i][j])
    """

    u_conv,v_conv = convection_2D(nx,ny,dt,dx,dy,u_i,v_i)

    u_diff,v_diff = diffusion_2D(vis,nx,ny,dt,dx,dy,u_i,v_i)

    u_mag,v_mag = magnetic_2D(rho,mu,nx,ny,dt,dx,dy,BX,BY,B_mag)

    for i in range(1,nx-1):
        for j in range(1,ny-1):

            u_i[i][j] = u_i[i][j] + u_conv[i][j] + u_diff[i][j]
            v_i[i][j] = v_i[i][j] + v_conv[i][j] + v_diff[i][j]

    #b,p_i = pressure_driving_force_2D(b,p_i,rho,nx,ny,p_nt,dt,dx,dy,u_i,v_i)

    #for i in range(1,nx-1):
        #for j in range(1,nx-1):

            #u_new[i][j] = u_i[i][j] - dt / rho * ((p_i[i + 1][j] - p_i[i - 1][j])
            #/ 2 * dx + (p_i[i][j + 1] - p_i[i][j - 1]) / 2 * dy)

            #v_new[i][j] = v_i[i][j] - dt / rho * ((p_i[i + 1][j] - p_i[i - 1][j])
            #/ 2 * dx + (p_i[i][j + 1] - p_i[i][j - 1]) / 2 * dy)

    u[n] = u_i.copy()
    v[n] = v_i.copy()
    #p[n] = p_i.copy()

for n in range(0,sim_t):
    for i in range(0,nx):
        for j in range(0,ny):

            U[n][i][j] = np.sqrt(u[n][i][j] ** 2 + v[n][i][j] ** 2)

#Meshgrid
X, Y = np.meshgrid(x, y)

#Plotting - 3D with z = velocity domain
def time_plot(r,t):
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, r[t][:], cmap=cm.viridis)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.show()

def time_arrow(t):
    plt.quiver(X,Y,u[t],v[t])
    plt.show()

def time_stream(t):
    plt.streamplot(X,Y,u[t],v[t])
    plt.show()
