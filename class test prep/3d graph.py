# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 00:39:20 2019

@author: Isaac
"""

#This will be used for 3-D plots

import numpy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import scipy
from scipy import integrate
from scipy import optimize

#x from 0 <= x <= 2 spacing 0.05
"x = numpy.linspace(0, 2, 41)"
x = numpy.arange(0, 2.05, 0.05)
#y from 0 <= y <= 1 spacing 0.05
"y = numpy.linspace(0, 1, 21)"
y = numpy.arange(0, 1.05, 0.05)

#setup the mesh grid
X, Y = numpy.meshgrid(x, y)

#setup z(x,y) to vary with x and y 
Z = numpy.sin(X**2 + Y**2)

#initialize the plot as a figure
fig = plt.figure()

#setup the plot in 3d space and the default angle
ax = fig.add_subplot(111, projection="3d")

#plot the surface with X, Y and Z and strides of 1 and using heatmap
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap = cm.hot)

#show plot
plt.show()

#2d
assert L == abs(L), "L must be an positive number"
assert L != 0, "L must be an positive number"
assert R == abs(R), "R must be an positive number"
assert R != 0, "R must be an positive number"
assert Fx == float(Fx), "Fx must be a float"
assert Fg == float(Fg), "Fg must be a float"
assert type(a) == np.ndarray, "a must be a list of floats"

#shooting
assert len(interval) == 2, "The interval should be a list of 2"
assert interval[1] > interval[0], "The interval's first element should be smaller than the seconds"
assert len(bcs) == 2, "The interval should be a list of 2"
assert N == abs(int(N)), "N must be an positive integer"
assert N != 0, "N must be an positive integer"

