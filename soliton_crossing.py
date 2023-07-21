
# -*- coding: utf-8 -*-
"""
Code uses a split-step algorithm to propagate two solitons acting under the
non-linear Schrodinger equation. This is then graphed on a 3D plot.
"""

import numpy as np
from scipy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt

dx = 0.025
L = 100  # length of spatial domain
N = L/dx  #number of discrete space steps
dt = 0.01 # size of time step
c = -1 # self interacting term coefficient
iterations = 1000
final_time = iterations*dt #final time
x = np.arange(-L/2, L/2, L/N)
k_space = (2*np.pi/L)*fftshift(np.arange(-N/2,N/2)) # k_space

y_1 = 2 / np.cosh(2*(x-4)) * np.exp(-1j*(x-4)) # first soliton wave multiplied
# by the exponetial term so that it moves
y_2 = 2 / np.cosh(2*(x+4)) * np.exp(1j*(x+4)) # second soliton
y_array_1 = y_1**2 ## initializes an array to store y values for each time step
y_array_2 = y_2**2

for i in range(int(final_time/dt)):

    #EXECUTES THE FIRST POTENTIAL STEP
    y_1 = np.exp(-0.5 * c * np.abs(y_1)**2 * dt * 1j, dtype=complex) * y_1
    y_2 = np.exp(-0.5 * c * np.abs(y_2)**2 * dt * 1j, dtype=complex) * y_2

    #FOURIER TRANSFORMS TO MOMENTUM SPACE
    y_1 = fft(y_1)
    y_2 = fft(y_2)

    #EXECUTES THE FIRST KINETIC STEP
    y_1 = np.exp(-0.5 * (k_space ** 2) * dt * 1j, dtype=complex) * y_1
    y_2 = np.exp(-0.5 * (k_space ** 2) * dt * 1j, dtype=complex) * y_2

    #INVERSE FOURIER TRANSFORMS TO MOMENTUM SPACE
    y_1 = ifft(y_1)
    y_2 = ifft(y_2)

    #EXECUTES THE SECOND POTENTIAL STEP
    y_1 = np.exp(-0.5 * c * np.abs(y_1)**2 * dt * 1j, dtype=complex) * y_1
    y_2 = np.exp(-0.5 * c * np.abs(y_2)**2 * dt * 1j, dtype=complex) * y_2

    y_array_1 = np.column_stack((y_array_1, np.abs(y_1)**2))
    y_array_2 = np.column_stack((y_array_2, np.abs(y_2)**2))


t_range = np.linspace(0, final_time+dt, int(final_time/dt)+1, endpoint=False)
T, X = np.meshgrid(t_range, x[int(N/2-N/20):int(N/2+N/20)])
Y_1 = y_array_1[int(N/2-N/20):int(N/2+N/20)]
Y_2 = y_array_2[int(N/2-N/20):int(N/2+N/20)]


fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_wireframe(X, T, Y_1, rstride=1000, cstride=100, color='red', linewidth=0.5)
ax.plot_wireframe(X, T, Y_2, rstride=1000, cstride=100, color='blue', linewidth=0.5)
ax.set_xlabel(r'Position/$z_{c}$')
ax.set_ylabel(r'Time/$t_{c}$')
ax.set_zlabel(r'Power/$P_{c}$')
ax.set_zlim(0,5.2)
ax.view_init(elev=30., azim=258)
plt.savefig('soliton_crossing_graph', dpi=600)