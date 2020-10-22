# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:44:18 2019

@author: Isaac
"""

import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import scipy
from scipy import integrate
from scipy import optimize


A = numpy.array([[1,0,0], [0,5,0], [0,0,1]])
b = numpy.array([1,2,3])

C = numpy.array([[1,2,3], [4,5,6], [7,8,9]])

#print(numpy.linalg.solve(A, b))
#print(numpy.linalg.matrix_power(C, 3))
#print(numpy.linalg.det(A))

t = numpy.linspace(1, 50, 50)
p = numpy.linspace(2, 51, 50)
#print(numpy.dot(t,t))
#print(t[::2])
#print(p[::2])

x = numpy.linspace(0, 2, 41)
y = numpy.linspace(0, 1, 21)

X, Y = numpy.meshgrid(x, y)
Z = numpy.sin(X**2 + Y**2)
fig = pyplot.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap = cm.hot)
pyplot.show()