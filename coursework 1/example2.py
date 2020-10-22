# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:37:45 2019

@author: Isaac
"""

from numpy import array, arange


def f(r):
    a, b, c, d = 1, 0.5, 0.5, 2
    x, y = r
    fx = a * x - b * x * y
    fy = c * x * y - d * y
    return array([fx, fy], float)

a = 0.0
b = 30.0
N = 1000
h = (b - a) / N

tpoints = arange(a, b, h)
xpoints = []
ypoints = []

r = array([2.0, 2.0], float)
for t in tpoints:
    xpoints.append(r[0])
    ypoints.append(r[1])
    k1 = h * f(r)
    k2 = h * f(r + 0.5 * k1)
    k3 = h * f(r + 0.5 * k2)
    k4 = h * f(r + k3)
    r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
