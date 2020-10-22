# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:12:23 2019

@author: iw1g17
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12,6)


def euler_method(f, x_end, y0, N):
    """Solve IVP y'=f(x, y(x)) on x \in [0, x_end] with y(0) = y0 using N+1 points, using Euler's method."""
    
    h = x_end / N
    x = np.linspace(0.0, x_end, N+1)
    
    y = np.zeros((N+1, len(y0)))
    y[0, :] = y0
    
    for n in range(N):
        
        y[n+1, :] = y[n, :] + h * f(x[n], y[n, :])
        
    return x, y

def bvector(x):
    """Simple function for Euler's method example"""
    
    return -np.sin(x)


print("Solution at x = 0.5 using h = 0.1 is y = {}.".format(y_5[-1, 0]))
print("Solution at x = 0.5 using h = 0.01 is y = {}.".format(y_50[-1, 0]))