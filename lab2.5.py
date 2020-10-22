# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:23:06 2019

@author: iw1g17
"""

import numpy
from matplotlib import pyplot as plt
import scipy
from scipy.integrate import quad
from scipy.integrate import odeint
from scipy.integrate import ode


#f = lambda y, x: -y
#x = numpy.linspace(0, 10)
#solivp = odeint(f, [1], x)

#def f1(x):
    #return numpy.sin(x)**2

#a = quad(f1, 0, numpy.pi)
#print(a)

x = numpy.linspace(0, 10)

def f1(s):
    return (numpy.sin(s))**2

b = quad(f1, 0, 10)

C = 1 + b

f = lambda y, x: -C*y
solivp = odeint(f, [1], x)