#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.animation as animation

# FFMPEG PATH :
# =============
#plt.rcParams['animation.ffmpeg_path'] = r"C:\Users\dlouapre\Documents\DATA\python\[ffmpeg]\bin\ffmpeg"
plt.rcParams['animation.ffmpeg_path'] = r'/Volumes/Data/Youtube/[ffmpeg]/ffmpeg'

# for complex number operations
from cmath import exp, pi, log, sqrt, phase

U0 = 1              # Velocity
a = 1               # Radius
b = 0.05            # Joukowski shift
c = 0.9           #        Joukowski deformation
alpha = 15/180*pi   # Angle of the flow

gamma = -4*pi*U0*(a+b)*np.sin(alpha)   # Circulation to get Kutta condition
theta = alpha                          # Global rotation so that the flow is horizontal
z0 = -b


def circlePot(z):
    w = complex(0,0)
    w += U0 * exp(-1j * alpha) * z
    w += U0 * exp(1j * alpha) * a**2 / z
    w += - 1j * gamma / (2 * pi) * log(z)
    return w

def rotTheta(z):
    return exp(-1j*theta)*z

def invTheta(z):
    return exp(1j*theta)*z


def Jouk(z):
    return z + c**2/z

def invJouk(zz):
    z = invTheta(zz)

    if z.real>=0:
        return z/2 + sqrt(z**2/4 - c**2)
    else:
        return z/2 - sqrt(z**2/4 - c**2)


# Choice of potential and inverse conformal mapping
f = circlePot
invConfMap = invJouk



# Range in X and Y    
W = 6
Y, X = np.mgrid[-W:W:100j, -W:W:100j]   #300
Z = X + 1j * Y



# Compute velocity using potential derivative
# Taking into account z0 shift and inverse conformal map
def complexVelocity(z):
    eps = 0.0001
    dwdz = (f(invConfMap(z+eps)-z0) - f(invConfMap(z)-z0))/eps
    return dwdz



U = np.zeros(Z.shape)
V = np.zeros(Z.shape)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):  
        dwdz = complexVelocity(Z[i,j])        
        U[i,j] = dwdz.real
        V[i,j] = -dwdz.imag

# Pressure deduced from velocity
P = - (U**2 + V**2) + U0**2




########
# PLOT #
########

# Plot
fig = plt.figure(figsize=(16,16),dpi=50)  #200
ax = fig.add_subplot(111)

# Seeds for the stream lines
K = 20   #80
seeds = np.array([[-0.99*W]*K, np.linspace(-0.99*W,0.99*W,K)])

# Optionnaly color 
col = np.sqrt(U**2 + V**2)
normalizer = matplotlib.colors.Normalize(0,2)

# col = "black"
# Density high enough to make sure all lines are ploted
ax.streamplot(X, Y, U, V, start_points = seeds.T, integration_direction = 'forward', 
              density=10, linewidth=2, color = col, norm=normalizer,cmap="jet", arrowstyle="-") # "seismic"
#ax.streamplot(X, Y, U, V, start_points = seeds.T, integration_direction = 'forward', 
#              density=10, linewidth=4, color="black", arrowstyle="-") # "seismic"


ax.set_xlim(-W,W)
ax.set_ylim(-W,W)

# Add a mask (at higher resolution for the shape of the wing)
Yr, Xr = np.mgrid[-W:W:400j, -W:W:400j]   #2000
Zr = Xr + 1j*Yr
mask = np.zeros(Zr.shape,dtype=bool)
level = np.zeros(Zr.shape)

for i in range(Zr.shape[0]):
    for j in range(Zr.shape[1]):        
        level[i,j] = abs(invTheta(invJouk(Zr[i,j]) + z0))
        #mask[i,j] = (abs(invTheta(invJouk(Zr[i,j]) + z0)) < 1.02*a) and (abs(invTheta(invJouk(Zr[i,j]) + z0)) > 0.98*a)
        mask[i,j] = (abs(invTheta(invJouk(Zr[i,j]) + z0)) < a)
mask = mask[:,::-1]
level = level[:,::-1]

#ax.imshow(level, extent=(-W, W, -W, W), alpha=1, cmap='jet', clim = (0,4), aspect='auto') 
ax.imshow(P[::-1,:], extent=(-W, W, -W, W), alpha=0.6, cmap='bwr', clim = (-2,2), aspect='auto')
immask = np.zeros((mask.shape[0], mask.shape[1],4),dtype=float)

for i in range(immask.shape[0]):
    for j in range(immask.shape[1]):
        if mask[i,j]:
            immask[i,j,3] = 1.0
        else:
            immask[i,j,3] = 0.0

ax.imshow(immask, extent=(-W, W, -W, W), aspect='auto')
       
# Trace the contour of the airfoil       
#xs = []
#ys = []
#for t in np.arange(0,2*pi,0.01):
#    z = z0 + a*exp(1j*t)            
#    Z = Jouk(z)
#    xs.append(Z.real)
#    ys.append(Z.imag)
#ax.plot(xs,ys,'--')
        
ax.set_aspect(1)

OUTPUT = True
name = "test"
NFRAMES = 300
FPS = 30
STEP = 50
DT = 0.001

particles = [np.r_[seeds[0,k],seeds[1,k]] for k in range(K)]

trajectories = [ax.plot(x[0],x[1],'or')[0] for x in particles]

def update(i):
    print("Frame "+str(i))
    for k in range(len(particles)):
        for s in range(STEP):
            zpart = particles[k][0] + 1j * particles[k][1]
            zvel = complexVelocity(zpart)
            particles[k][0] += zvel.real * DT
            particles[k][1] += - zvel.imag * DT
        trajectories[k].set_data(particles[k][0],particles[k][1])

ani = animation.FuncAnimation(fig, update, frames = NFRAMES)

if OUTPUT:    
    writer = animation.FFMpegWriter(fps=FPS, bitrate = None)
    ani.save(name+".mp4", writer = writer, savefig_kwargs={'facecolor':'black'})
else:
    plt.show() 


#plt.savefig("test.png")