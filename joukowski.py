#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm

from cmath import exp, pi, log, sqrt



U0 = 1              # Velocity
a = 1               # Radius
c = 0.9            # Joukowski deformation

z0 = complex(-0.05,0.2)

def Jouk(z):
    return (z+z0) + c**2/(z+z0)

# Trace the contour of the airfoil       


rs = np.arange(0.88,0.92,0.01)

K = rs.shape[0]

cmap = cm.jet

cols = [cmap(int(i)) for i in [np.floor((float(k)/(K-1)*256)) for k in range(K)]]

Zs = [[Jouk(r*exp(1j*t)) for t in np.arange(0,2*pi,0.01)] for r in rs]

fig, ax = plt.subplots(figsize=(9,9))
[ax.plot([z.real for z in Z],[z.imag for z in Z],'--',color = c) for Z,c in zip(Zs,cols)]

ax.plot([z.real for z in [Jouk(exp(1j*t)) for t in np.arange(0,2*pi,0.01)]],[z.imag for z in [Jouk(exp(1j*t)) for t in np.arange(0,2*pi,0.01)]],'-',color = "black")

ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_aspect(1)
plt.show()