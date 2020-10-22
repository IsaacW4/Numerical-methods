# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:59:51 2019

@author: iw1g17
"""

import numpy
from matplotlib import pyplot as plt
import scipy
from scipy.integrate import quad
from scipy.integrate import odeint
from scipy.integrate import ode


v = (1.0 + 1e-4 * numpy.random.rand(8))
#print(v)

A = numpy.diag(v)
B = numpy.diag(v, -1)
#print(A)

C = numpy.reshape(v, (2,4))
#print(C)
D = numpy.reshape(v, (4,2))
#print(D)
E = numpy.reshape(v, (2,2,2))
#print(E)

def f1(x):
    return numpy.sin(x)**2

a = quad(f1, 0, numpy.pi)
#print(a)

def f2(x):
    return numpy.exp(-x**2 * (numpy.cos(2*numpy.pi*x))**2)

b = quad(f2, 0, 1)
#print(b)

f3 = lambda x: numpy.exp(-x**2 * (numpy.cos(2*numpy.pi*x))**2)
c = quad(f3, 0, 1)
#print(c)

f4_tol = quad(f2, 0, 1, epsabs = 1e-14, epsrel = 1e-14)
#print(f4_tol)



f = lambda y, x: -y
x = numpy.linspace(0, 10)
#solivp = odeint(f, [1], x)

g = lambda y, x: -numpy.cos(y)
z = numpy.linspace(0, 10)
solveivp = odeint(g, [1], z, )    #atol=1 #Changes the accuracy of the solveivp function 

#plt.plot(solveivp, z)







