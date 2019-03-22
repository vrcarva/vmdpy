# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 19:50:28 2019

@author: Vinícius Rezende Carvalho

Test script for Variational Mode Decomposition (Dragomiretskiy and Zosso, 2014)
Original paper:
Dragomiretskiy, K. and Zosso, D. (2014) ‘Variational Mode Decomposition’, 
IEEE Transactions on Signal Processing, 62(3), pp. 531–544. doi: 10.1109/TSP.2013.2288675.
original MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/44765-variational-mode-decomposition
"""
#from __future__ import division# if python 2
import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD

# Time Domain 0 to T
T = 1000
fs = 1/T
t = np.arange(1,T+1)/T
freqs = 2*np.pi*(t-0.5-fs)/(fs)

# center frequencies of components
f_1 = 2
f_2 = 24
f_3 = 288

# modes
v_1 = (np.cos(2*np.pi*f_1*t))
v_2 = 1/4*(np.cos(2*np.pi*f_2*t))
v_3 = 1/16*(np.cos(2*np.pi*f_3*t))

#
#% for visualization purposes
fsub = {1:v_1,2:v_2,3:v_3}
wsub = {1:2*np.pi*f_1,2:2*np.pi*f_2,3:2*np.pi*f_3}

# composite signal, including noise

f = v_1 + v_2 + v_3 + 0.1*np.random.randn(v_1.size)
f_hat = np.fft.fftshift((np.fft.fft(f)))

# some sample parameters for VMD
alpha = 2000       # moderate bandwidth constraint
tau = 0.            # noise-tolerance (no strict fidelity enforcement)
K = 3              # 3 modes
DC = 0             # no DC part imposed
init = 1           # initialize omegas uniformly
tol = 1e-7


# Run actual VMD code

u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)

#%%
# Simple Visualization of decomposed modes


plt.figure()
plt.plot(u.T)
plt.title('Decomposed modes')


# For convenience here: Order omegas increasingly and reindex u/u_hat
sortIndex = np.argsort(omega[-1,:])
omega = omega[:,sortIndex]
u_hat = u_hat[:,sortIndex]
u = u[sortIndex,:]
linestyles = ['b', 'g', 'm', 'c', 'c', 'r', 'k']

fig1 = plt.figure()
plt.subplot(411)
plt.plot(t,f)
plt.xlim((0,1))
for key, value in fsub.items():
    plt.subplot(4,1,key+1)
    plt.plot(t,value)
fig1.suptitle('Original input signal and its components')


fig2 = plt.figure()
plt.loglog(freqs[T//2:], abs(f_hat[T//2:]))
plt.xlim(np.array([1,T/2])*np.pi*2)
ax = plt.gca()
ax.grid(which='major', axis='both', linestyle='--')
fig2.suptitle('Input signal spectrum')


fig3 = plt.figure()
for k in range(K):
    plt.semilogx(2*np.pi/fs*omega[:,k], np.arange(1,omega.shape[0]+1), linestyles[k])
fig3.suptitle('Evolution of center frequencies omega')


fig4 = plt.figure()
plt.loglog(freqs[T//2:], abs(f_hat[T//2:]), 'k:')
plt.xlim(np.array([1, T//2])*np.pi*2)
for k in range(K):
    plt.loglog(freqs[T//2:], abs(u_hat[T//2:,k]), linestyles[k])
fig4.suptitle('Spectral decomposition')
plt.legend(['Original','1st component','2nd component','3rd component'])


fig4 = plt.figure()

for k in range(K):
    plt.subplot(3,1,k+1)
    plt.plot(t,u[k,:], linestyles[k])
    plt.plot(t, fsub[k+1], 'k:')

    plt.xlim((0,1))
    plt.title('Reconstructed mode %d'%(k+1))




