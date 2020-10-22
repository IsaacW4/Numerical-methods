# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:58:26 2019

@author: iw1g17
"""

import numpy as np
import scipy
from scipy.integrate import solve_ivp, solve_bvp
from matplotlib import pyplot as plt
plt.rc('font', size=16)




def task1(L, R, fx, fg, theta0):
    """
       ---Inputs---
       L - Float, length of the hair
       R - Float, radius of the spherical head
       fx - Float, force from wind acting on each hair in the positive x direction
       fg - Float, force from gravity acting on each hair in the negative z direction
       theta0 - List, specifies the list of values where hair meets head
       
       ---Returns---
       (x,z) - Coordinates, of the hairs on the head
    
    """
    theta = np.linspace(0, 2*np.pi, 100)
   
    x1 = R*np.cos(theta)
    x2 = R*np.sin(theta)
    plt.plot(x1,x2)


    plt.xlim(-15,15)
    plt.ylim(-15,15)
 
    plt.grid(linestyle='--')
    plt.title('How to plot a circle with matplotlib ?', fontsize=8)
 
   
    for i in theta[0:51:5]:
        s = np.linspace(0, 4, 100)
        x = (s+10)*np.cos(i)
        y = (s+10)*np.sin(i)
        plt.plot(x, y)
       
    plt.show()

    
       
    
    #PDEs 3(a,b)
    def shooting_dirichlet(f, interval, bcs, N):
        
        """
    Solve a BVP using shooting
    
    Parameters
    ----------
    
    f: function
        RHS function defining BVP: y'' = f(x, y, y')
    interval: list of two floats
        [a, b] where BVP defined on x \in [a, b]
    bcs: list of two floats
        Defines Dirichlet boundary values [y(a), y(b)]
    N: single integer
        Number of interior points in the grid, used for output only
    
    Returns
    -------
    
    x: array of floats
        Location of grid points
    y: array of floats
        Solution at grid points
        """
    
        s = np.linspace(interval[0], interval[1], N+2)
        y = np.zeros_like(s)
        y[0] = bcs[0]
    
        def rhs_shooting(s, z):
            """
            RHS function for shooting (z = [y, y']; solve z' = [z[1], f(...)])
        
            Parameters
            ----------
        
                x: float
                Location
                z: list of floats
                [y, y']
        
            Returns
            -------
        
            dzdx: list of floats
                [y', y''] = [z[1], f(x, z[0], z[1])]
                """
                
            v = z[1]
            theta = z[0]
            dtheta = v
            dvbyds = s * fg * np.cos(theta) + s * fx * np.sin(theta)
            #y, dy = z
            return np.array([dtheta, dvbyds])
    
        def residuals_shooting(guess):
            """
            Solve IVP given y'(a): compute r = y(b) - bcs[1]
        
            Parameters
            ----------
        
            guess: float
                Guess for derivative of solution at x=a
        
            Returns
            -------
        
            residual: float
               Values of residual r at x=b
               """
            sol_shoot = solve_ivp(rhs_shooting, interval, [bcs[0], guess])
            return sol_shoot.y[1, -1] - bcs[1]
    
    
        sol = scipy.optimize.root_scalar(residuals_shooting, method='brentq', bracket=[-10, 10])
        guess = sol.root
        sol_shoot = solve_ivp(rhs_shooting, interval, [bcs[0], guess], t_eval=s)
        y_all = sol_shoot.y
        return s, y_all[0, :]
    
        #Solve IVP with theta 
        dxds = np.cos(y_all)
        dzds = np.sin(y_all)
        x0 = R*np.cos(theta0)
        z0 = R*np.sin(theta0)
    
        x = solve_ivp(dxds, interval, x0)
        z = solve_ivp(dzds, interval, z0)
        
        
    
    
    


    