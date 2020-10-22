# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 10:36:09 2019

@author: Isaac
"""

#This will be made to manipulate matrices in python e.g. multiplying / evecs and evals

import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import scipy
from scipy import integrate
from scipy import optimize

#Buidling matrices within python, use numpy arrays

#3x3 matrix
A = numpy.array([[1,7,2], [3,5,0], [6,8,1]])

#3x1 matrix
b = numpy.array([5,2,7])

#calculating A^3
c = numpy.linalg.matrix_power(A, 3)

#solving the problem Ax = b and solving for x 
d = numpy.linalg.solve(A, b)

#calculate lower triangular of a matrix
v = numpy.tril(A, -1)

#calculating determinant of A
e = numpy.linalg.det(A)

#creating a vector of 40 entries from 1 to 2
t = numpy.linspace(1, 2, 40)

#calculating the dot product
f = numpy.dot(t, t)
w = t**2

#summing odd values 
k = numpy.sum(w[::2])

#linalg.eig gives 2 returns, evecs and evals 
evals = numpy.linalg.eigvals(A - 2 * numpy.eye(3))


h = numpy.min(numpy.abs(evals))
g = numpy.max(numpy.abs(evals))

#defining a function
def f(x):
    return x**2

#finding a root of the equation
root = optimize.newton(f, 1.0)

#integrate a function 
a1, b1 = integrate.quad(f, 0, 2)

pp = numpy.array(1)
print(pp)


