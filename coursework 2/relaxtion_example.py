# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:31:56 2019

@author: iw1g17
"""

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
def p(x):
    return np.ones_like(x)
def q(x):
    return np.zeros_like(x)
def f(x):
    return -np.ones_like(x)
y_bc = [0.0, 1.0]

def y_exact(x):
    return 2.0 * np.exp(1.0) / (np.exp(1.0) - 1.0) * (1.0 - np.exp(-x)) - x

N = 25
x = np.linspace(0.0, 1.0, N+2)
h = x[1] - x[0]
A_diag = h**2 * q(x[1:-1]) - 2.0
A_subdiag = 1.0 - h / 2.0 * p(x[2:-1]) 
A_supdiag = 1.0 + h / 2.0 * p(x[1:-2]) 

A = diags([A_subdiag, A_diag, A_supdiag], [-1, 0, 1], shape=(N,N))
b = h**2 * f(x[1:-1])
b[0] -= y_bc[0] * (1.0 - h / 2.0 * p(x[1]))
b[-1] -= y_bc[1] * (1.0 + h / 2.0 * p(x[-2]))

y_int = spsolve(A, b)
y = np.hstack((0.0, y_int, 1.0))
plt.plot(x, y, 'kx')
plt.show()