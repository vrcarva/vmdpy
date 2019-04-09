# Variational mode decomposition Python Package

Function for calculating Variational Mode Decomposition (Dragomiretskiy and Zosso, 2014) of a signal  
Original VMD paper:  
Dragomiretskiy, K. and Zosso, D. (2014) ‘Variational Mode Decomposition’, 
IEEE Transactions on Signal Processing, 62(3), pp. 531–544. doi: 10.1109/TSP.2013.2288675.

original MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/44765-variational-mode-decomposition  


## Installation 

1) Dowload the project from https://github.com/vrcarva/vmdpy, then run "python setup.py install" from the project folder

OR

2) pip install vmdpy

## Citation and Contact
If you find this package useful, we kindly ask you to cite it in your work.   
Vinicius Carvalho (2019-), Variational Mode Decomposition in Python  

A paper will soon be submitted and linked here.  

contact: vrcarva@ufmg.br  
Vinícius Rezende Carvalho  
Programa de Pós-Graduação em Engenharia Elétrica – Universidade Federal de Minas Gerais, Belo Horizonte, Brasil  
Núcleo de Neurociências - Universidade Federal de Minas Gerais  


## Example script
```python
#%% Simple example  
import numpy as np  
import matplotlib.pyplot as plt  
from vmdpy import VMD  

#. Time Domain 0 to T  
T = 1000  
fs = 1/T  
t = np.arange(1,T+1)/T  
freqs = 2*np.pi*(t-0.5-fs)/(fs)  

#. center frequencies of components  
f_1 = 2  
f_2 = 24  
f_3 = 288  

#. modes  
v_1 = (np.cos(2*np.pi*f_1*t))  
v_2 = 1/4*(np.cos(2*np.pi*f_2*t))  
v_3 = 1/16*(np.cos(2*np.pi*f_3*t))  

f = v_1 + v_2 + v_3 + 0.1*np.random.randn(v_1.size)  

#. some sample parameters for VMD  
alpha = 2000       # moderate bandwidth constraint  
tau = 0.            # noise-tolerance (no strict fidelity enforcement)  
K = 3              # 3 modes  
DC = 0             # no DC part imposed  
init = 1           # initialize omegas uniformly  
tol = 1e-7  


#. Run actual VMD code  
u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)  
```