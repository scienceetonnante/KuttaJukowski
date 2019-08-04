#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
from numpy import cos, sin, arctan2
import matplotlib.animation as animation

# FFMPEG PATH :
# =============
#plt.rcParams['animation.ffmpeg_path'] = r"C:\Users\dlouapre\Documents\DATA\python\[ffmpeg]\bin\ffmpeg"
plt.rcParams['animation.ffmpeg_path'] = r'/Volumes/Data/Youtube/[ffmpeg]/ffmpeg'


N = 500

XMIN = -8
XMAX = 8
YMIN = -4.5
YMAX = 4.5
V0 = 1
WIND = 1
L = 0.1


# Initial self avoiding positions
positions = np.zeros((N,2))
for i in range(N):
    print("Placing particle" + str(i))
    fnd = False
    while not fnd:
        positions[i,0] = uniform(XMIN, XMAX)
        positions[i,1] = uniform(YMIN, YMAX)
        fnd = True
        for j in range(i):
            r = positions[i,:] - positions[j,:]
            if(r[0]**2 + r[1]**2 < L**2):
                fnd = False

winds = WIND * np.vstack((np.ones(N),np.zeros(N))).T

# Initial angle
angles = np.r_[uniform(0,2*np.pi,N)]
                            
fig = plt.figure(figsize=(16,9),dpi=200)
ax = fig.add_subplot(111)

line, = plt.plot(positions[:,0],positions[:,1],'o')
plt.tight_layout()

ax.set_xlim(XMIN, XMAX)
ax.set_ylim(YMIN, YMAX)
ax.set_aspect(1)


OUTPUT = True
name = "kinetic_wind"
NFRAMES = 6000
FPS = 30
DT = 0.01

def update(f):
    print("Frame "+str(f))
    velocities = V0 * np.vstack((cos(angles),sin(angles))).T + winds
    positions[:,:] = positions[:,:] + DT * velocities
    
    for i in range(N):
        for j in range(i+1,N):
            r = positions[i,:] - positions[j,:]
            if(r[0]**2 + r[1]**2 < L**2):
                angles[i] = arctan2(r[1],r[0])
                angles[j] = np.pi + angles[i]
            
    for i in range(N):
        if(positions[i,0]<XMIN):
            # angles[i] = np.pi - angles[i]
            positions[i,0] = XMAX - (XMIN - positions[i,0])
        if(positions[i,0]>XMAX):
            #angles[i] = np.pi - angles[i]
            positions[i,0] = XMIN + (positions[i,0] - XMAX)
        if(positions[i,1]<YMIN):
            positions[i,1] = YMIN + (YMIN - positions[i,1])
            angles[i] = 2*np.pi - angles[i]
        if(positions[i,1]>YMAX):
            positions[i,1] = YMAX - (positions[i,1] - YMAX)
            angles[i] = 2*np.pi - angles[i]
            
    
    line.set_data(positions[:,0],positions[:,1])


ani = animation.FuncAnimation(fig, update, frames = NFRAMES)

if OUTPUT:    
    writer = animation.FFMpegWriter(fps=FPS, bitrate = None)
    ani.save(name+".mp4", writer = writer)
else:
    plt.show() 

#plt.savefig("kinetic.png")
#plt.show()