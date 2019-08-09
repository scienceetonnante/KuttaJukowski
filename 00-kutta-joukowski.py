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
from cmath import exp, pi, log, sqrt

name = "kutta jukowski circulation"

U0 = 1              # Velocity
a = 1               # Radius
b1 = 0.05           # Joukowski shift real
b2 = 0.2            # Joukowski shift imag
c = 0.9             # Joukowski deformation
alpha = 20/180*pi   # Angle of the flow


# Circulation to get Kutta condition
gamma = -4*pi*U0*((a+b1) * np.sin(alpha) + b2*np.cos(alpha))
# Shift of circle  as a complex number
z0 = -b1 + 1j*b2                         


# Complex potential around a circle with attack angle alpha
def circlePot(z):
    w = complex(0,0)
    w += U0 * exp(-1j * alpha) * z
    w += U0 * exp(1j * alpha) * a**2 / z
    w += - 1j * gamma / (2 * pi) * log(z)
    return w


# Conformal map as a composition of translation, Joukowski and rotation
def Jouk(z):
    return - exp(1j*alpha) * ((z+z0) + c**2/(z+z0))


# Inverse conformal map 
# There are two roots of the Joukowksi map, we return the root with larger modulus
# To be outside of the original circle

def invJouk(zz):
    
    z = exp(1j*alpha)*zz

    r1 = z/2 + sqrt(z**2/4 - c**2) - z0
    r2 = z/2 - sqrt(z**2/4 - c**2) - z0

    if abs(r1) > abs(r2):
        return r1
    else:        
        return r2

# Compute velocity using potential derivative
# Taking into account z0 shift and inverse conformal map
def complexVelocity(z):
    eps = 0.0001
    dwdz = (circlePot(invJouk(z+eps)) - circlePot(invJouk(z)))/eps
    return dwdz


###############
# CALCULATION #
###############    

# Range in X and Y    
W = 4
Y, X = np.mgrid[-W:W:30j, -W:W:30j]
Z = X + 1j * Y


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
fig = plt.figure(figsize=(16,16),dpi=300)
ax = fig.add_subplot(111)

# Seeds for the stream lines
K = 60
seeds = np.array([[-0.99*W]*K, np.linspace(-0.99*W,0.99*W,K)])

# Optionnaly color 
col = np.sqrt(U**2 + V**2)
normalizer = matplotlib.colors.Normalize(0,2)

# col = "black"
# Density high enough to make sure all lines are ploted
#ax.streamplot(X, Y, U, V, start_points = seeds.T, integration_direction = 'forward', 
#              density=10, linewidth=2, color = col, norm=normalizer,cmap="jet", arrowstyle="-")
#ax.streamplot(X, Y, U, V, start_points = seeds.T, integration_direction = 'forward', 
#              density=10, linewidth=2, color="slategray", arrowstyle="-") # "seismic"

ax.quiver(X,Y,U - U0,V)

ax.set_xlim(-W,W)
ax.set_ylim(-W,W)

# Add a mask (at higher resolution for the shape of the wing)

Yr, Xr = np.mgrid[-W:W:3000j, -W:W:3000j]
Zr = Xr + 1j*Yr

immask = np.zeros((Zr.shape[0], Zr.shape[1],4),dtype=float)

for i in range(immask.shape[0]):
    for j in range(immask.shape[1]):
        if abs(invJouk(Zr[i,j])) < a:
            immask[i,j,3] = 1.0
        else:
            immask[i,j,3] = 0.0

ax.imshow(P[::-1,:], extent=(-W, W, -W, W), alpha=0.6, cmap='bwr', clim = (-2,2), aspect='auto')
ax.imshow(immask[::-1,:], extent=(-W, W, -W, W), aspect='auto')

       
## Trace the contour of the airfoil       
#xs = []
#ys = []
#for t in np.arange(0,2*pi,0.01):
#    z = a*exp(1j*t)            
#    Z = exp(-2*1j*alpha)*Jouk(z)
#    xs.append(-Z.real)
#    ys.append(-Z.imag)
#ax.plot(xs,ys,'--')
#        
#ax.set_aspect(1)



#OUTPUT = True
#
#NFRAMES = 450
#FPS = 30
#STEP = 30
#DT = 0.001
#
#particles = [np.r_[seeds[0,k],seeds[1,k]] for k in range(K)]
#
#trajectories = [ax.plot(x[0],x[1],'o',color="magenta",markersize=12)[0] for x in particles]
#
#
#def update(i):
#    print("Frame "+str(i))
#    for k in range(len(particles)):
#        for s in range(STEP):
#            zpart = particles[k][0] + 1j * particles[k][1]
#            zvel = complexVelocity(zpart)
#            particles[k][0] += zvel.real * DT
#            particles[k][1] += - zvel.imag * DT
#        trajectories[k].set_data(particles[k][0],particles[k][1])
#
#ani = animation.FuncAnimation(fig, update, frames = NFRAMES)
#
#if OUTPUT:    
#    writer = animation.FFMpegWriter(fps=FPS, bitrate = None)
#    ani.save(name+".mp4", writer = writer)
#else:
#    plt.show() 


plt.savefig(name+".png")