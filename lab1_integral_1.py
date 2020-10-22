# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:07:56 2019

@author: iw1g17
"""

def integral_1(Nstrips):
    width = 1/Nstrips 
    integral=0
    for anything in range(Nstrips):
        height = (anything / Nstrips)**2
        integral = integral + width * height
        
    return integral

print("One Hundred strips: ", integral_1(100))