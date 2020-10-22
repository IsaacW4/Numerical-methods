# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:06:36 2019

@author: iw1g17
"""

import numpy
import scipy

A = numpy.array([[1,2,3], [4,5,6], [7,8,9]])
b = numpy.array([1, 3, 6])
x = numpy.linalg.solve(A, b)
print(x)

lam, v = numpy.linalg.eig(A)
print(lam[0])
print(v[:, 0])