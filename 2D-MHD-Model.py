from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from decimal import *

#Get filename
filename = os.path.basename(__file__)
filename = filename.split('.')[0]

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

            b[i][j] = -1 * ((rho / dt) * ((u_i[i + 1][j] - u_i[i - 1][j]) / (2 * dx)
             + (v_i[i][j + 1] - v_i[i][j - 1]) / (2 * dy)))

    for n in range(p_nt):
        for i in range(1,nx-1):
            for j in range(1,ny-1):

                p_i[i][j] = (((p_i[i + 1][j] - p_i[i - 1][j]) * dy ** 2  + (p_i[i][j + 1]
                 - p_i[i][j - 1]) * dx ** 2 - (b[i][j] * dx ** 2 * dy ** 2)) /
                 (2 * (dx ** 2 + dy ** 2)))

    #p_i[0, :] = 0
    #p_i[nx-1, :] = 0
    #p_i[:, 0] = 0
    #p_i[:, ny-1] = 0

    return b, p_i

def convection_2D(nx,ny,dt,dx,dy,u_i,v_i):

    # CHRIS: VECTORISED - NumPy roll functions slide arrays in different directions.
    # done in the underlying C/Fortran NumPy code and so are much faster than nested
    # for loops.

    u_conv = -u_i*(u_i-np.roll(u_i,1,axis=0))*dt/dx - v_i*(u_i-np.roll(u_i,1,axis=1))*dt/dy

    v_conv = -u_i*(v_i-np.roll(v_i,1,axis=0))*dt/dx - v_i*(v_i-np.roll(v_i,1,axis=1))*dt/dy

    # ignore the edges: since the above code will have modified the edge cells, reset them.
    u_conv[:,0] = 0
    u_conv[:,-1] = 0
    u_conv[0,:] = 0
    u_conv[-1,:] = 0
    v_conv[:,0] = 0
    v_conv[:,-1] = 0
    v_conv[0,:] = 0
    v_conv[-1,:] = 0

    return u_conv, v_conv

def diffusion_2D(vis,nx,ny,dt,dx,dy,u_i,v_i):

    # CHRIS: VECTORISED - NumPy roll functions slide arrays in different directions.
    # done in the underlying C/Fortran NumPy code and so are much faster than nested
    # for loops.

    u_diff = (vis*dt/dx**2) * (np.roll(u_i,1,axis=0)-2*u_i +
    np.roll(u_i,-1,axis=0)) + (vis*dt/dy**2) * (np.roll(u_i,1,axis=1)
    -2*u_i + np.roll(u_i,-1,axis=1))

    v_diff = (vis*dt/dx**2) * (np.roll(v_i,1,axis=0)-2*v_i +
    np.roll(v_i,-1,axis=0)) + (vis*dt/dy**2) * (np.roll(v_i,1,axis=1)
    -2*v_i + np.roll(v_i,-1,axis=1))

    # ignore the edges: since the above code will have modified the edge cells, reset them.
    u_diff[:,0] = 0
    u_diff[:,-1] = 0
    u_diff[0,:] = 0
    u_diff[-1,:] = 0
    v_diff[:,0] = 0
    v_diff[:,-1] = 0
    v_diff[0,:] = 0
    v_diff[-1,:] = 0

    return u_diff ,v_diff

def magnetic_2D(rho,mu,nx,ny,dt,dx,dy,BX,BY,B_mag):

    u_mag = (((BX * dt) / (rho * mu * dx)) * (BX - np.roll(BX,1,axis=0))) +(((BY * dt) / (rho * mu * dy)) * (BX - np.roll(BX,1,axis=1))) +((dt /(2 * rho * mu * dx)) * ((B_mag) ** 2 - (np.roll(B_mag,1,axis=0)) ** 2))

    v_mag = (((BX * dt) / (rho * mu * dx)) * (BY - np.roll(BY,1,axis=0))) +(((BY * dt) / (rho * mu * dy)) * (BY - np.roll(BY,1,axis=1))) + ((dt /(2 * rho * mu * dy)) * ((B_mag) ** 2 - (np.roll(B_mag,1,axis=1)) ** 2))

    return u_mag, v_mag

#from poisson_solver import pressure_driving_force_2D

#Options
class options:

    def __init__(self):
        self.IncludeConvection = bool(True)
        self.IncludeDiffusion = bool(True)
        self.IncludeMagnetism = bool(False)
        self.CalculatePressure = bool(False)
        self.CalculateVelocityMagnitude = bool(False)

opt = options()

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
nt = 5001
p_nt = 101 #peudo-time for pressure
dx = (x_max - x_min) / (nx-1)
dy = (y_max - y_min) / (ny-1)
dt = (t_end - t_initial) / (nt-1)
cfl_max=0.75

#Control size of perturbation down interface
amp = 2
freq = 2

#Control actual number of time steps simulation runs over
sim_t = 1001

#Create arrays
x = np.zeros(nx)
y = np.zeros(ny)
t = np.zeros(nt)
u = np.zeros((sim_t,nx,ny))
u_temp = np.zeros((nx,ny))
u_i = np.zeros((nx,ny))
u_new = np.zeros((nx,ny))
v = np.zeros((sim_t,nx,ny))
v_temp = np.zeros((nx,ny))
v_i = np.zeros((nx,ny))
v_new = np.zeros((nx,ny))
U = np.zeros((sim_t,nx,ny))
#p = np.zeros((sim_t,nx,ny))

#Populate dimensional vectors
x = np.linspace(x_min,x_max,num=nx)
y = np.linspace(y_min,y_max,num=ny)
t = np.linspace(t_initial,t_end,num=nt)

#Calculate magentic field components throughout domain
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

u_new = u[0].copy()
v_new = v[0].copy()

#Meshgrid
X, Y = np.meshgrid(x, y)

#Iterate through time over spaital (x,y) domain from IC's
for n in range(1,sim_t):

    # CHRIS: Check CFL limit
    cfl = np.max((u_i/dx + v_i/dy)*dt)
    if cfl > cfl_max:
        print('[' + str(filename) + '] Exceeding CFL limit of {} with CFL={}'.format(cfl_max,cfl))

    if n % 10 == 0:
        print('[' + str(filename) + '] Step {}/{} t={} CFL={}'.format(n,sim_t,t[n],cfl))

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

    if opt.IncludeConvection is True:
        u_conv,v_conv = convection_2D(nx,ny,dt,dx,dy,u_i,v_i)
    else:
        u_conv = 0
        v_conv = 0

    if opt.IncludeDiffusion is True:
        u_diff,v_diff = diffusion_2D(vis,nx,ny,dt,dx,dy,u_i,v_i)
    else:
        u_diff = 0
        v_diff = 0

    if opt.IncludeMagnetism is True:
        u_mag,v_mag = magnetic_2D(rho,mu,nx,ny,dt,dx,dy,BX,BY,B_mag)
    else:
        u_mag = 0
        v_mag = 0

    # CHRIS: VECTORISED - can just add the arrays - NumPy will do it internally.
    u_i = u_i + u_conv + u_diff + u_mag
    v_i = v_i + v_conv + v_diff + v_mag

    if opt.CalculatePressure is True:
        b,p_i = pressure_driving_force_2D(rho,nx,ny,p_nt,dt,dx,dy,u_i,v_i)

        for i in range(1,nx-1):
            for j in range(1,nx-1):

                u_new[i][j] = u_i[i][j] - dt / rho * ((p_i[i + 1][j] - p_i[i - 1][j])
                / 2 * dx + (p_i[i][j + 1] - p_i[i][j - 1]) / 2 * dy)

                v_new[i][j] = v_i[i][j] - dt / rho * ((p_i[i + 1][j] - p_i[i - 1][j])
                / 2 * dx + (p_i[i][j + 1] - p_i[i][j - 1]) / 2 * dy)

        u[n] = u_new.copy()
        v[n] = v_new.copy()
        p[n] = p_i.copy()

    else:

        u[n] = u_i.copy()
        v[n] = v_i.copy()

    # CHRIS: Do a visual update every 100 timesteps
    if n % 100==0:

        plt.clf()
        plt.streamplot(X,Y,u_i,v_i)
        plt.title('$t$ = ' + str(n/(nt-1)))
        plt.draw()
        plt.pause(0.001)

if opt.CalculateVelocityMagnitude is True:
    # CHRIS: VECTORISED
    U = np.sqrt(u**2 + v**2)

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
