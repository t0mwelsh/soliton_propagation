# -*- coding: utf-8 -*-
"""
Uses the split-step algorithm to graph how a wave propagates with time. y (the
the inital wave shape) and c should be varied to see different phenomena.
"""

import numpy as np
from scipy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt

dx = 0.025
L = 100  # length of spatial domain
N = L/dx  #number of discrete space steps
dt = 0.0025 # size of time step
c = 0 # self interacting term coefficient (can be altered)
final_time = 1000*dt #final time
x = np.arange(-L/2, L/2, L/N)
k_space = (2*np.pi/L)*fftshift(np.arange(-N/2,N/2)) # k_space

y = 2*np.exp(-x**2) # initial wave shape
y_array = y**2 ## initializes an array to store y values for each time step

for i in range(int(final_time/dt)):

    #EXECUTES THE FIRST POTENTIAL STEP
    y = np.exp(-0.5 * c * np.abs(y)**2 * dt * 1j, dtype=complex) * y

    #FOURIER TRANSFORMS TO MOMENTUM SPACE
    y = fft(y)

    #EXECUTES THE FIRST KINETIC STEP
    y = np.exp(-0.5 * (k_space ** 2) * dt * 1j, dtype=complex) * y

    #INVERSE FOURIER TRANSFORMS TO MOMENTUM SPACE
    y = ifft(y)

    #EXECUTES THE SECOND POTENTIAL STEP
    y = np.exp(-0.5 * c * np.abs(y)**2 * dt * 1j, dtype=complex) * y

    y_array = np.column_stack((y_array, np.square(np.abs(y))))

space_divider = 25 # relates how much of the total space propagated we want graphed
t_range = np.linspace(0, final_time+dt, int(final_time/dt)+1, endpoint=False)
T, X = np.meshgrid(t_range, x[int(N/2-N/space_divider):int(N/2+N/space_divider)])
Y = y_array[int(N/2-N/space_divider):int(N/2+N/space_divider)]

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_wireframe(X, T, Y, rstride=10, cstride=50)
ax.set_xlabel(r'Position/$z_{c}$')
ax.set_ylabel(r'Time/$t_{c}$')
ax.set_zlabel(r'Power/$P_{c}$')
#ax.set_title('Amplitude Decay')
plt.savefig('c=-1_decay', dpi=600)