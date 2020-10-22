# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 00:31:20 2019

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
"x = numpy.linspace(1, 4, 301)"
x = numpy.arange(1.0, 4.01, 0.01)

#y values of a function dependent on x 
y = numpy.log(x) * numpy.sin(2 * numpy.pi * x)

#initializing plot
pyplot.plot(x,y)

#x label
pyplot.xlabel("$x$")

#y label
pyplot.ylabel("$\log(x) \sin(2 \pi x)$")

#title 
pyplot.title("Figure for question 7")

#show plot
pyplot.show()