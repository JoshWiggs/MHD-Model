from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from decimal import *

#Options object
class options:

    def __init__(self):
        self.IncludeConvection = bool(True)
        self.IncludeDiffusion = bool(True)
        self.IncludeMagnetism = bool(False)
        self.CalculatePressure = bool(False)
        self.CalculateVelocityMagnitude = bool(True)
        self.FTCSMethod = bool(False) #Select either this or Leap Frog method
        self.LeapFrogMethod = bool(True) #if using this ensure diffusion is on

        #Advanced Options
        self.TemporalSparseStorage = bool(False) # not compatible with leapfrog
        #TODO: add sparse storage modes

    class setup:

        def __init__(self):
            self.x_min = -2
            self.x_max = 2
            self.y_min = -2
            self.y_max = 2
            self.t_initial = 0
            self.t_end = 2.0
            self.viscosity = 0.1 #use 0.1 for stable FTCS
            self.density = 1 #1
            self.magnetic_permeability = 1 #1
            self.nx = 100
            self.ny = 100
            self.nt = 2001
            self.simulation_run_time = 2001
            #TODO: add initial condition setup (maybe new class)

    class pressure:

        def __init__(self):
            self.pressure_smoothing_iterations = 101

    class perturbation:

        def __init__(self):
            self.amplitude = 2
            self.frequency = 2
            #TODO: add functions to allow control of perturbation form

    class debug:

        def __init__(self):
            self.cfl_max = 0.75

#Assign objects
opt = options()
su = options.setup()
per = options.perturbation()
db = options.debug()
pres = options.pressure()

#Fetch filename
filename = os.path.basename(__file__)
filename = filename.split('.')[0]

def magnetic_field_2D(nx,ny,x,y):

    B_x = np.zeros(nx)
    B_y = np.zeros(ny)
    B_mag = np.zeros((nx,ny))

    for i in range(0,nx):

        B_x[i] = 2 * np.abs(y[i])

    for j in range(0,ny):

        B_y[j] = 2 * np.abs(x[j])

    BX,BY = np.meshgrid(B_x,B_y)

    B_mag = np.sqrt(BX**2 + BY**2)

    return B_x, B_y, B_mag

def pressure_driving_force_2D(rho,nx,ny,p_nt,dt,dx,dy,u_i,v_i):

    p_i = np.zeros((nx,ny))

    b = -1 * ((rho / dt) * ((np.roll(u_i,-1,axis=0) - np.roll(u_i,1,axis=0)) /
    (2 * dx) + (np.roll(v_i,-1,axis=1) - np.roll(v_i,1,axis=1)) / (2 * dy)))

    for n in range(p_nt):

        p_i = (((np.roll(p_i,-1,axis=0) - np.roll(p_i,1,axis=0)) * dy ** 2  +
        (np.roll(p_i,-1,axis=1) - np.roll(p_i,1,axis=1)) * dx ** 2 -
        (b * dx ** 2 * dy ** 2)) / (2 * (dx ** 2 + dy ** 2)))

        p_i[:,0] = 0
        p_i[:,-1] = 0
        p_i[0,:] = 0
        p_i[-1,:] = 0

    # ignore the edges: since the above code will have modified the edge cells, reset them.
    b[:,0] = 0
    b[:,-1] = 0
    b[0,:] = 0
    b[-1,:] = 0


    return b, p_i

def convection_2D(dt,dx,dy,u_i,v_i):

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

def diffusion_2D(vis,dt,dx,dy,u_i,v_i):

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

def magnetic_2D(vis,mu,dt,dx,dy,BX,BY,B_mag):

    u_mag = (((BX * dt) / (vis * mu * dx)) * (BX - np.roll(BX,1,axis=0))) +(((BY * dt) / (vis * mu * dy)) * (BX - np.roll(BX,1,axis=1))) + ((dt /(2 * vis * mu * dx)) * ((B_mag) ** 2 - (np.roll(B_mag,1,axis=0)) ** 2))

    v_mag = (((BX * dt) / (vis * mu * dx)) * (BY - np.roll(BY,1,axis=0))) +(((BY * dt) / (vis * mu * dy)) * (BY - np.roll(BY,1,axis=1))) + ((dt /(2 * vis * mu * dy)) * ((B_mag) ** 2 - (np.roll(B_mag,1,axis=1)) ** 2))

    # ignore the edges: since the above code will have modified the edge cells, reset them.
    u_mag[:,0] = 0
    u_mag[:,-1] = 0
    u_mag[0,:] = 0
    u_mag[-1,:] = 0
    v_mag[:,0] = 0
    v_mag[:,-1] = 0
    v_mag[0,:] = 0
    v_mag[-1,:] = 0

    return u_mag, v_mag

def leapfrog_convection_2D(alpha,dt,dx,dy,u_i,v_i):

    u_conv = alpha * (-u_i*(np.roll(u_i,-1,axis=0)-np.roll(u_i,1,axis=0))*dt/dx - v_i*(np.roll(u_i,-1,axis=1)-np.roll(u_i,1,axis=1))*dt/dy)

    v_conv = alpha * (-u_i*(np.roll(v_i,-1,axis=0)-np.roll(v_i,1,axis=0))*dt/dx - v_i*(np.roll(v_i,-1,axis=1)-np.roll(v_i,1,axis=1))*dt/dy)

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

def leapfrog_diffusion_2D(alpha,vis,dt,dx,dy,u_i,v_i,u_ii,v_ii):

    u_diff = alpha * ((2*vis*dt/dx**2) * (np.roll(u_i,1,axis=0)-u_ii + np.roll(u_i,-1,axis=0)) + (2*vis*dt/dy**2) * (np.roll(u_i,1,axis=1) -u_ii + np.roll(u_i,-1,axis=1)))

    v_diff = alpha * ((2*vis*dt/dx**2) * (np.roll(v_i,1,axis=0)-v_ii + np.roll(v_i,-1,axis=0)) + (2*vis*dt/dy**2) * (np.roll(v_i,1,axis=1) -v_ii + np.roll(v_i,-1,axis=1)))

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

def leapfrog_magnetic_2D(alpha,vis,mu,dt,dx,dy,BX,BY,B_mag):

    u_mag = alpha * ((((BX * dt) / (vis * mu * dx)) * (np.roll(BX,-1,axis=0) - np.roll(BX,1,axis=0))) +(((BY * dt) / (vis * mu * dy)) * (np.roll(BX,-1,axis=1) - np.roll(BX,1,axis=1))) +((dt /(2 * vis * mu * dx)) * ((np.roll(B_mag,-1,axis=0)) ** 2 - (np.roll(B_mag,1,axis=0)) ** 2)))

    v_mag = alpha * ((((BX * dt) / (vis * mu * dx)) * (np.roll(BY,-1,axis=0) - np.roll(BY,1,axis=0))) +(((BY * dt) / (vis * mu * dy)) * (np.roll(BY,-1,axis=1) - np.roll(BY,1,axis=1))) +((dt /(2 * vis * mu * dy)) * ((np.roll(B_mag,-1,axis=1)) ** 2 - (np.roll(B_mag,1,axis=1)) ** 2)))

    # ignore the edges: since the above code will have modified the edge cells, reset them.
    u_mag[:,0] = 0
    u_mag[:,-1] = 0
    u_mag[0,:] = 0
    u_mag[-1,:] = 0
    v_mag[:,0] = 0
    v_mag[:,-1] = 0
    v_mag[0,:] = 0
    v_mag[-1,:] = 0

    return u_mag, v_mag

#Calculate grid resolutions
dx = (su.x_max - su.x_min) / (su.nx-1)
dy = (su.y_max - su.y_min) / (su.ny-1)
dt = (su.t_end - su.t_initial) / (su.nt-1)

#Create arrays
x = np.zeros(su.nx)
y = np.zeros(su.ny)
t = np.zeros(su.nt)
u_temp = np.zeros((su.nx,su.ny))
u_i = np.zeros((su.nx,su.ny))
v_temp = np.zeros((su.nx,su.ny))
v_i = np.zeros((su.nx,su.ny))

if opt.TemporalSparseStorage is True:
    u = np.zeros((int(su.simulation_run_time/100) + 1,su.nx,su.ny))
    v = np.zeros((int(su.simulation_run_time/100) + 1,su.nx,su.ny))
else:
    u = np.zeros((su.simulation_run_time,su.nx,su.ny))
    v = np.zeros((su.simulation_run_time,su.nx,su.ny))

#Open variable for velocity magnitude if CalculateVelocityMagnitude option is on
if opt.CalculateVelocityMagnitude is True:
    U = np.zeros((su.simulation_run_time,su.nx,su.ny))

#Open variables for pressure if CalculatePressure option is on
if opt.CalculatePressure is True:
    u_new = np.zeros((su.nx,su.ny))
    v_new = np.zeros((su.nx,su.ny))
    p_nt = pres.pressure_smoothing_iterations #peudo-time for pressure
    p = np.zeros((su.simulation_run_time,su.nx,su.ny))

#Open additional time variable if leapfrog method being used
if opt.LeapFrogMethod is True:
    u_ii = np.zeros((su.nx,su.ny)) #u^n-1 for leapfrog
    alpha = 1 / (1 + ((2*su.viscosity*dt) * ((dx**2) + (dy**2))/((dx**2)*(dy**2))))

#Populate dimensional vectors
x = np.linspace(su.x_min,su.x_max,num=su.nx)
y = np.linspace(su.y_min,su.y_max,num=su.ny)
t = np.linspace(su.t_initial,su.t_end,num=su.nt)

#Calculate magentic field components throughout domain
B_x, B_y, B_mag = magnetic_field_2D(su.nx,su.ny,x,y)

BX,BY = np.meshgrid(B_x,B_y)

#IC's
"""
u[0][:][:] = 0.25
v[0][:][:] = 0.25

for i in range(0,24):
    for j in range(0,24):
        u[0][24+i][24+j] = 2
        v[0][24+i][24+j] = 2

"""

for i in range(0,int(np.round((su.nx / 2.0)))):
     u[0][:][i] = -0.1
     u[0][:][int((np.round(su.nx / 2.0))) + i] = 0.1

     xi = np.zeros(su.nx)
     for i in range(0,su.nx):
         xi[i] = per.amplitude * (np.sin((np.pi * (per.frequency * x[i]))))

     xi[0:int(np.round((su.nx/4.0)))] = 0
     xi[int(np.round((su.nx*0.75))):su.nx] = 0

     v[0][:][int(np.floor(su.nx/2.0)) - 1] = xi
     v[0][:][int(np.floor(su.nx/2.0))] = xi
     v[0][:][int(np.floor(su.nx/2.0)) + 1] = xi


u_i = u[0].copy()
v_i = v[0].copy()

#Set initial conditions in additional velocity variables in CalculatePressure option is on
if opt.CalculatePressure is True:
    u_new = u[0].copy()
    v_new = v[0].copy()


#Meshgrid
X, Y = np.meshgrid(x, y)

#Pressure IC set up
if opt.CalculatePressure is True:

    b,p_i = pressure_driving_force_2D(su.density,su.nx,su.ny,p_nt,dt,dx,dy,u_i,v_i)

    p[0] = p_i.copy()

#For Leap Frog first time step needs to be done with FTCS,
#that is done here
if opt.LeapFrogMethod is True:
    #If IncludeConvection option is turned on then
    if opt.IncludeConvection is True:
        u_conv,v_conv = convection_2D(dt,dx,dy,u_i,v_i)
    else:
        u_conv = 0
        v_conv = 0

    #If IncludeDiffusion option is turned on then
    if opt.IncludeDiffusion is True:
        u_diff,v_diff = diffusion_2D(su.viscosity,dt,dx,dy,u_i,v_i)
    else:
        u_diff = 0
        v_diff = 0

    #If IncludeMagnetism option is turned on then
    if opt.IncludeMagnetism is True:
        u_mag,v_mag = magnetic_2D(su.viscosity,su.magnetic_permeability,dt,dx,dy,BX,BY,B_mag)
    else:
        u_mag = 0
        v_mag = 0

    # CHRIS: VECTORISED - can just add the arrays - NumPy will do it internally.
    u_i = u_i + u_conv + u_diff + u_mag
    v_i = v_i + v_conv + v_diff + v_mag

    u_ii = u[0].copy()
    v_ii = v[0].copy()
    u[1] = u_i.copy()
    v[1] = v_i.copy()

#Iterate through time over spaital (x,y) domain from IC's
if opt.LeapFrogMethod is True:
    initial_step = int(2)
else:
    initial_step = int(1)

for n in range(initial_step,su.simulation_run_time):

    # CHRIS: Check CFL limit
    cfl = np.max((u_i/dx + v_i/dy)*dt)
    if cfl > db.cfl_max:
        print('[{}] Exceeding CFL limit of {} with CFL={}'.format(filename,db.cfl_max,cfl))

    if n % 10 == 0:
        print('[{}] Step {}/{} t={} CFL={}'.format(filename,n,su.simulation_run_time,t[n],cfl))

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

    #If IncludeConvection option is turned on then
    if opt.IncludeConvection is True:
        if opt.FTCSMethod is True:
            u_conv,v_conv = convection_2D(dt,dx,dy,u_i,v_i)
        elif opt.LeapFrogMethod is True:
            u_conv,v_conv = leapfrog_convection_2D(alpha,dt,dx,dy,u_i,v_i)
    else:
        u_conv = 0
        v_conv = 0

    #If IncludeDiffusion option is turned on then
    if opt.IncludeDiffusion is True:
        if opt.FTCSMethod is True:
            u_diff,v_diff = diffusion_2D(su.viscosity,dt,dx,dy,u_i,v_i)
        elif opt.LeapFrogMethod is True:
            u_diff,v_diff = leapfrog_diffusion_2D(alpha,su.viscosity,dt,dx,dy,u_i,v_i,u_ii,v_ii)
    else:
        u_diff = 0
        v_diff = 0

    #If IncludeMagnetism option is turned on then
    if opt.IncludeMagnetism is True:
        if opt.FTCSMethod is True:
            u_mag,v_mag = magnetic_2D(su.viscosity,su.magnetic_permeability,dt,dx,dy,BX,BY,B_mag)
        elif opt.LeapFrogMethod is True:
            u_mag,v_mag = leapfrog_magnetic_2D(alpha,su.viscosity,su.magnetic_permeability,dt,dx,dy,BX,BY,B_mag)
    else:
        u_mag = 0
        v_mag = 0

    # CHRIS: VECTORISED - can just add the arrays - NumPy will do it internally.
    if opt.FTCSMethod is True:
        u_i = u_i + u_conv + u_diff + u_mag
        v_i = v_i + v_conv + v_diff + v_mag
    elif opt.LeapFrogMethod is True:
        u_i = (alpha * u_ii) + u_conv + u_diff + u_mag
        v_i = (alpha * v_ii) + v_conv + v_diff + v_mag

    if opt.CalculatePressure is True:
        b,p_i = pressure_driving_force_2D(su.density,su.nx,su.ny,p_nt,dt,dx,dy,u_i,v_i)

        u_new = u_i - dt / su.density * ((np.roll(p_i,-1,axis=0) - np.roll(p_i,1,axis=0))
        / 2 * dx + (np.roll(p_i,-1,axis=1) - np.roll(p_i,1,axis=1)) / 2 * dy)

        v_new = v_i - dt / su.density * ((np.roll(p_i,-1,axis=0) - np.roll(p_i,1,axis=0))
        / 2 * dx + (np.roll(p_i,-1,axis=1) - np.roll(p_i,1,axis=1)) / 2 * dy)

        if n % 100==0:

            plt.clf()
            plt.subplot(2,1,1)
            plt.title('$t$ = ' + str(n/(su.nt-1)))
            plt.streamplot(X,Y,u_i,v_i)
            plt.subplot(2,1,2)
            plt.title('$t$ = ' + str(t[n]))
            plt.contourf(X,Y,p_i,cmap=cm.seismic)
            plt.colorbar(label = '$p$')
            plt.draw()
            plt.pause(0.001)

        if opt.TemporalSparseStorage is True:
            if n % 100 == 0:
                u[int(n/100)] = u_new.copy()
                v[int(n/100)] = v_new.copy()
                p[int(n)] = p_i.copy()
        elif opt.LeapFrogMethod is True:
            u_ii = u[n - 1].copy()
            v_ii = v[n - 1].copy()
            u[n] = u_new.copy()
            v[n] = v_new.copy()
            p[n] = p_i.copy()
        else:
            u[n] = u_new.copy()
            v[n] = v_new.copy()
            p[n] = p_i.copy()

    else:

        if n % 100==0:

            plt.clf()
            plt.streamplot(X,Y,u_i,v_i)
            plt.title('$t$ = ' + str(t[n]))
            plt.draw()
            plt.pause(0.001)

        if opt.TemporalSparseStorage is True:
            if n % 100 == 0:
                u[int(n/100)] = u_i.copy()
                v[int(n/100)] = v_i.copy()
        elif opt.LeapFrogMethod is True:
            u_ii = u[n - 1].copy()
            v_ii = v[n - 1].copy()
            u[n] = u_i.copy()
            v[n] = v_i.copy()
        else:
            u[n] = u_i.copy()
            v[n] = v_i.copy()

    # CHRIS: Do a visual update every 100 timesteps


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
