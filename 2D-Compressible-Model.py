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
nx = 51
ny = 51
nt = 1001
p_nt = 101 #peudo-time for pressure
dx = (x_max - x_min) / (nx-1)
dy = (y_max - y_min) / (ny-1)
dt = (t_end - t_initial) / (nt-1)

#Create arrays
x = np.zeros(nx)
y = np.zeros(ny)
t = np.zeros(nt)
u = np.ones((nt,nx,ny))
u_temp = np.ones((nx,ny))
u_i = np.ones((nx,ny))
u_new = np.ones((nx,ny))
v = np.ones((nt,nx,ny))
v_temp = np.ones((nx,ny))
v_i = np.ones((nx,ny))
v_new = np.ones((nx,ny))
U = np.zeros_like(u)
b = np.zeros_like(u_i)
p = np.zeros_like(u)
rho = np.ones_like(u)
p_i = np.zeros_like(u_i)
rho_i = np.ones_like(u_i)

#Populate dimensional vectors
x = np.linspace(x_min,x_max,num=nx)
y = np.linspace(y_min,y_max,num=ny)
t = np.linspace(t_initial,t_end,num=nt)

#IC's
for i in range(10,25):
    for j in range(10,25):
        u[0][i][j] = 2
        v[0][i][j] = 2

u_i = u[0].copy()
v_i = v[0].copy()


def continuity_eqn_solver(rho_i,nx,ny,dt,dx,dy,u_i,v_i):

    for i in range(1,nx-1):
        for j in range(1,ny-1):

            rho_i[i][j] = rho_i[i][j] - ((rho_i[i][j] * dt / dx) * (u_i[i][j] -
            u_i[i - 1][j])) - ((rho_i[i][j] * dt / dy) * (v_i[i][j] -
            v_i[i][j-1])) - ((u_i[i][j] * dt / dx) * (rho_i[i][j] -
            rho_i[i - 1][j])) - ((v_i[i][j] * dt /dy) * (rho_i[i][j] -
            rho_i[i][j - 1]))

    return rho_i

#Poisson Solver
def pressure_driving_force_2D(b,p_i,rho_i,nx,ny,p_nt,dt,dx,dy,u_i,v_i):

    for i in range(1,nx-1):
        for j in range(1,ny-1):

            b[i][j] = ((rho_i[i][j] / dt) * ((u_i[i + 1][j] - u_i[i - 1][j]) / (2 * dx)
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

#Iterate through time over spaital (x,y) domain from IC's
for n in range(1,101):

    if n % 10 == 0:
        print(n)

    u_temp = u_i.copy()
    v_temp = v_i.copy()

    rho_i = continuity_eqn_solver(rho_i,nx,ny,dt,dx,dy,u_i,v_i)

    for i in range(1,nx-1):
        for j in range(1,ny-1):

            u_i[i][j] = u_temp[i][j] - ((dt / dx) * u_temp[i][j] * (u_temp[i][j]
             - u_temp[i - 1][j])) - ((dt / dy) * v_temp[i][j] * (u_temp[i][j] -
             u_temp[i][j - 1]))

            v_i[i][j] = v_temp[i][j] - ((dt / dx) * u_temp[i][j] * (v_temp[i][j]
             - v_temp[i - 1][j])) - ((dt / dy) * v_temp[i][j] * (v_temp[i][j] -
             v_temp[i][j - 1]))

    b,p_i = pressure_driving_force_2D(b,p_i,rho_i,nx,ny,p_nt,dt,dx,dy,u_i,v_i)

    for i in range(1,nx-1):
        for j in range(1,nx-1):

            u_new[i][j] = u_i[i][j] - dt / rho_i[i][j] * ((p_i[i + 1][j] - p_i[i - 1][j])
            / 2 * dx + (p_i[i][j + 1] - p_i[i][j - 1]) / 2 * dy)

            v_new[i][j] = v_i[i][j] - dt / rho_i[i][j] * ((p_i[i + 1][j] - p_i[i - 1][j])
            / 2 * dx + (p_i[i][j + 1] - p_i[i][j - 1]) / 2 * dy)

    u[n] = u_i.copy()
    v[n] = v_i.copy()
    rho[n] = rho_i.copy()
    p[n] = p_i.copy()

for n in range(0,nt):
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
