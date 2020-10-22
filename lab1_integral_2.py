# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:12:35 2019

@author: iw1g17
"""

from lab1_integral_1 import integral_1

def integral_4(Nstrips):
    width = 1/Nstrips
    integral = 0 
    for anything in range(Nstrips):
        height = (anything / Nstrips)
        integral = integral + width * height 
    return integral 

def integral_total(Nstrips):
    return integral_4(Nstrips) + integral_1(Nstrips)