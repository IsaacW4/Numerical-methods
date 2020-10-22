# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 08:27:58 2019

@author: Isaac
"""

#This will be used to setup 2-D plots 

import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import scipy
from scipy import integrate
from scipy import optimize



#x values from 1 <= x <= 4 spacing h=0.01
x = numpy.linspace(0, 10, 100)
x1 = numpy.sin(x)

#y values of a function dependent on x 
y = numpy.linspace(0, 2, 60)
y1 = numpy.exp(-y**2)


#initializing plot
pyplot.plot(x,x1, "md")

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
