# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:26:51 2019

@author: iw1g17
"""

import matplotlib.pyplot as plt
import numpy as np 

x = np.linspace(0, 1, 80)
y = np.linspace(0, 2, 60)

s = np.array(np.sin(x))
t = np.array(np.exp(-y**2))

plt.plot(s,t)
plt.show()

