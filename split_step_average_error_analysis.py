# -*- coding: utf-8 -*-
"""
Code investigates the accuracy of the split-step algorithm due to the size of
time-step. It achieves this by propagating a soliton that shouldn't move in
time with the algorithm and then measuring how much it does move compared to
its initial state. This dependency is then graphed with a plot of the residuals
below.
"""

import numpy as np
from scipy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt

dx = 0.05 # space step
L = 100  # length of spatial domain
N = L/dx  #number of discrete space steps
dt = 0.0025 # size of starting time step
iterations = 1000 # (the final time will then be dt*iterations)
c = -1 # self interacting term coefficient

x = np.arange(-L/2, L/2, L/N) # position space
k_space = (2*np.pi/L)*fftshift(np.arange(-N/2,N/2)) # k_space
divisor = 50 # a variable that is used to decide how many of the iterations
# we want to store. Storing less is more efficient but lower accuracy

errors = np.empty(2)
time_step_points = 25

for j in range(1, time_step_points):
    h = dt*j # effective time-step as we increase the time-step
    y = 2 / np.cosh(2*x) #starting condition. Needs to be this be this so that
    # theoretically it should not move
    y_array = y ## initializes an array to store y values for each time step
    error_curve = np.empty(int(N))

    for i in range(iterations):

        #EXECUTES THE FIRST POTENTIAL STEP
        y = np.exp(-0.5 * c * np.abs(y)**2 * h * 1j, dtype=complex) * y

        #FOURIER TRANSFORMS TO MOMENTUM SPACE
        y = fft(y)

        #EXECUTES THE FIRST KINETIC STEP
        y = np.exp(-0.5 * (k_space ** 2) * h * 1j, dtype=complex) * y

        #INVERSE FOURIER TRANSFORMS TO MOMENTUM SPACE
        y = ifft(y)

        #EXECUTES THE SECOND POTENTIAL STEP
        y = np.exp(-0.5 * c * np.abs(y)**2 * h * 1j, dtype=complex) * y

        if i%int(iterations/divisor) == 0:
            y_array = np.column_stack((y_array, np.abs(y)))
            error_curve = np.vstack((error_curve, y_array[:,-1] - y_array[:,0]))
            # looking at the difference between the inital condition and final

    error_curve = np.delete(error_curve, (0), axis=0) #coming from np.empty
    error_curve = np.average(error_curve, axis=0)

    difference_squared = np.sqrt(np.sum(np.square(error_curve)))
    errors = np.vstack((errors, [h, difference_squared]))

errors = np.delete(errors, (0), axis=0)

x_points, y_points = np.log(errors[:,0]), np.log(errors[:,1])

[a, b], [residuals], _, _, _ = np.polyfit(x_points, y_points, 1, full=True)

residues = y_points - (a*x_points + b)

fig=plt.figure()

ax = fig.add_subplot(211)
ax.plot(x_points, a*x_points + b, color='orange')
ax.scatter(x_points, y_points, marker="x")
ax.set_xlabel(r'log(dt/$t_{c}$)', fontsize=14)
ax.set_ylabel(r'log(error/$z_{c}$)', fontsize=14)
ax.grid()

ax_2 = fig.add_subplot(212)
ax_2.plot(x_points, y_points-y_points, color='orange')
ax_2.scatter(x_points, residues, marker="x")
ax_2.set_xlabel(r'log(dt/$t_{c}$)', fontsize=14)
ax_2.set_ylabel(r'log(error/$z_{c}$)', fontsize=14)
ax_2.grid()

fig.align_ylabels()
plt.savefig('split_step_error_analysis_graph', dpi=600, bbox_inches='tight')

print(residuals)







