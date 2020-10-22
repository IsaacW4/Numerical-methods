# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:55:25 2019

@author: Isaac
"""

import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import scipy
from scipy import integrate
from scipy import optimize



#x values from 1 <= x <= 4 spacing h=0.01
x = numpy.linspace(-10, 10, 100)


#y values of a function dependent on x 
y = numpy.linspace(-10, 10, 100)

z = (x**2 + y**2 -1)**3 - x**2 * y**3

#initializing plot
pyplot.plot(z,"md")

#x label
pyplot.xlabel("$x$", size=23)

#y label
pyplot.ylabel("$\log(x) \sin(2 \pi x)$", size=23)

#title 
pyplot.title("Figure for question 7")

pyplot.legend(["A simple line"])

#show plot
pyplot.show()

z = numpy.linspace(0, 1, 6)
g = numpy.sum(z[::2])