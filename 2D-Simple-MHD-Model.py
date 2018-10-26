import numpy as np
import time, sys
import matplotlib.pyplot as plt

# 2D MHD Model Variables

#ICs
v=0.01

#Resolution
nx = 100
ny = 100
nt = 50

#Domain (doubled for spatial domains)
lx = 2
ly = 1
lt = 5

#Inital time
t0 = 0
x0 = 0

#Step size
dx = lx / nx
dy = (ly * 2) / ny
dt = (lt) / nt

#Open arrays
x = np.zeros(nx)
y = np.zeros(ny)
t = np.zeros(nt)

#Set first array value
x[0] = x0
y[0] = (ly * -1)
t[0] = t0

#Populate arrays
for i in range(1,nx):
    x[i] = x[i-1] + dx

for j in range(1,ny):
    y[j] = y[j-1] + dy

for k in range(1,nt):
    t[k] = t[k-1] + dt

#Create fluids

Fx = x.copy()

iy = ny / 2
iy = int(iy)
F1y = np.zeros(iy)
F2y = np.zeros(iy)

for i in range(0,iy):

    F1y[i] = y[i]
    F2y[i] = y[i + 50]

#Meshgrid

F1 = np.meshgrid(Fx,F1y)
F2 = np.meshgrid(Fx,F2y)

#Add time domians

F1_temp = np.empty([50 , 2 , 50 , 100]) #todo: generalise
F2_temp = np.empty([50 , 2 , 50 , 100])

for i  in range(0,nt):
    F1_temp[i] = F1

'''
F1_temp = [None] * nt
F2_temp = [None] * nt

for i in range(0,nt):
    F1_temp[i]= F1
    F2_temp[i]= F2

F1 = F1_temp.copy()
F2 = F2_temp.copy()
'''
#Movement

#for i in range(1,nt):
#        F1[i][0][:][:] = F1[i - 1][0][:][:] + v

#Plotting

#plt.plot(F2[0],F2[1],'+',color='red')
#plt.plot(F1[0],F1[1],'+',color='blue')
#plt.show()
