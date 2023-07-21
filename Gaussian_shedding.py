# -*- coding: utf-8 -*-
"""
Code uses the split-step algorithm to propagate two Gaussian curves coming
towards each other. The program then plots snapshots of these interactions at
different times showing a decomposition of the Gaussians into stable solitons
crossing through each other.
"""

import numpy as np
from scipy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt

dx = 0.05
L = 100
c = -1
iterations = 1000
divisor = 50
dt = 0.01

def IC(x): # two Gaussians going towards each other
    return 2*np.exp(-(x-5)**2 - 1j*(x-7)) + 2*np.exp(-(x+5)**2 + 1j*(x+7))

def split_step(dx, L, dt, c, iterations, IC, divisor):
# function performs the split-step algorithm to output a set of coordinates of
# the initial condition after it has propagated to the final time

    N = int(L/dx)  #number of discrete space steps
    x = np.arange(-L/2, L/2, L/N)
    k = (2*np.pi/L)*fftshift(np.arange(-N/2,N/2)) # k_space
    y = IC(x)
    y_array = y**2 ## initializes an array to store y values for each time step

    for i in range(1, iterations+1):

        #EXECUTES THE FIRST POTENTIAL STEP
        y = np.exp(-0.5 * c * np.abs(y)**2 * dt * 1j, dtype=complex) * y

        #FOURIER TRANSFORMS TO MOMENTUM SPACE
        y = fft(y)

        #EXECUTES THE FIRST KINETIC STEP
        y = np.exp(-0.5 * (k ** 2) * dt * 1j, dtype=complex) * y

        #INVERSE FOURIER TRANSFORMS TO MOMENTUM SPACE
        y = ifft(y)

        #EXECUTES THE SECOND POTENTIAL STEP
        y = np.exp(-0.5 * c * np.abs(y)**2 * dt * 1j, dtype=complex) * y

        if i%int(iterations/divisor) == 0:
            y_array = np.column_stack((y_array, np.abs(y)**2))

    return x, y_array

x, y_array = split_step(dx, L, dt, c, iterations, IC, divisor)

time_array = np.array([1, 7, 21, 39])

fig = plt.figure()

for i in time_array:
    index = np.where(time_array==i)[0][0] + 1
    ax = fig.add_subplot(2,2, index)
    ax.plot(x, y_array[:,i])
    ax.text(-14.5, 8, f"t = {dt*i}"+r'/$t_{c}$', fontsize=11)
    #plt.legend(loc="upper left")
    #ax.grid()
    if index > 2:
        ax.set_xlabel(r'Position/$z_{c}$')
    if index % 2 != 0:
        ax.set_ylabel(r'Power/$P_{c}$')
    ax.set_xlim(-15,15)
    ax.set_yscale('log')
    ax.set_ylim(10**-2,17)

plt.tight_layout()
plt.savefig('Gaussian shedding figure', dpi=600)
