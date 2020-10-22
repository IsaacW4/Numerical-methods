# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 19:16:04 2019

@author: Isaac
"""
L = 4
R = 10

import numpy as np
import scipy
from scipy.integrate import solve_ivp, odeint
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rc('font', size=16)

"""Why I picked the Shooting method
------------------------------------
I picked the shooting method for this problem over other solvers because,
while not being as easy as a black box solver for the problem, it doesn't completely
rely on in-built scipy solvers in order to solve the BVP, instead we change the problem to an IVP
and solve it that way using an initial guess that as the user has control over. So I believe
using shooting gives a balance between using solvers and having control over the algorithm, moreover,
I believe that shooting is an easier method than relaxation to impose, so my initial view on the problem
was to either use shooting or the black box solvers.
"""


#Defining all the different equations needed for the 2D and 3D hair plotting model
#Originally used the below method for defining functions, but, throughout the problem, was just easier to define functions
#g = lambda y, x: -numpy.cos(y)
#The below functions are strictly used for the 2D case
def d2theta(s, theta, dtheta, phi, dphi, fg, fx):
    return s*fg*np.cos(theta)
 
def d2phi(s, theta, dtheta, phi, dphi, fg, fx):
    return s*fx*np.cos(phi)*np.sin(theta)
 
def equation3a(s,theta,dtheta,fg,fx):
    return s*fg*np.cos(theta)+s*fx*np.sin(theta)
 
def dx(theta):
    return np.cos(theta)
 
def dz(theta):
    return np.sin(theta)

#These extra 3 functions are used in the 3D case, most notably we're allowing phi to vary in this case
def dx1(theta, phi):
    return np.cos(theta)*np.cos(phi)

def dy(theta, phi):
    return -np.cos(theta)*np.sin(phi)

def dz1(theta):
    return np.sin(theta)

#The function used to carry out the shooting process
def shooting_dirichlet(d2theta, interval, bcs, N, fg, fx):
    """
    Solve a BVP using shooting
   
    Parameters
    ----------
    
    d2theta: function
        RHS function defining BVP: y'' = f(x, y, y')
    interval: list of two floats
        [a, b] where BVP defined on x \in [a, b]
    bcs: list of two floats
        Defines Dirichlet boundary values [y(a), y(b)]
    N: single integer
        Number of interior points in the grid, used for output only
    fg: integer
        Force from gravity
    fx: integer
        Force from wind
   
    Returns
    -------
   
    x: array of floats
        Location of grid points
    z: array of floats
        Solution at grid points
    """
    
    assert len(interval) == 2, "The interval should be a list of 2"
    assert interval[1] > interval[0], "The interval's first element should be smaller than the seconds"
    assert len(bcs) == 2, "The interval should be a list of 2"
    assert N == abs(int(N)), "N must be an positive integer"
    assert N != 0, "N must be an positive integer"
    assert L == abs(L), "L must be an positive number"
    assert L != 0, "L must be an positive number"
    assert R == abs(R), "R must be an positive number"
    assert R != 0, "R must be an positive number"
    assert fx == float(fx), "fx must be a float"
    assert fg == float(fg), "fg must be a float"
    
    
    
    def rhs_shooting(s, z):
        """
        RHS function for shooting (z = [y, y']; solve z' = [z[1], f(...)])
     
        Parameters
        ----------
       
        s: float
        Location
        z: list of floats
        [y, y']
       
        Returns
        -------
       
        Array: An array of floats
        """
        #Building an array to start the shooting method
        theta, dtheta = z
        return np.array([dtheta, d2theta(s, theta, dtheta, fg, fx)])
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
        #Solving the IVP for the guess of the solution with our first BCs
        shoot = solve_ivp(rhs_shooting, interval, [bcs[0], guess])
        return shoot.y[1, -1] - bcs[1]
    
    #Finding a root of the residuals function using a bracketing interval and the brentq method
    sol = scipy.optimize.brentq(residuals_shooting, -10, 10)
    #Defining the list of values to define s on
    slists = np.linspace(interval[0], interval[1], N+2)
    #Solving the last IVP for finding theta values for given s values
    sol_shoot = solve_ivp(rhs_shooting, interval, [bcs[0], sol], t_eval=slists)
    return slists, sol_shoot.y
   
    
#The same function as above, trying to build it into a 3D version
def shooting_dirichlet1(d2theta, d2phi, interval, bcs, N, fg, fx):
    """
    Solve a BVP using shooting
   
    Parameters
    ----------
    
    d2theta: function
        RHS function defining BVP: y'' = f(x, y, y')
    d2phi: function
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
    
    assert len(interval) == 2, "The interval should be a list of 2"
    assert interval[1] > interval[0], "The interval's first element should be smaller than the seconds"
    assert N == abs(int(N)), "N must be an positive integer"
    assert N != 0, "N must be an positive integer"
    assert L == abs(L), "L must be an positive number"
    assert L != 0, "L must be an positive number"
    assert R == abs(R), "R must be an positive number"
    assert R != 0, "R must be an positive number"
    assert fx == float(fx), "fx must be a float"
    assert fg == float(fg), "fg must be a float"
    
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
        theta, dtheta, phi, dphi = z
        return np.array([dtheta, d2theta(s, theta, dtheta, phi, dphi, fg, fx), dphi, d2phi(s, theta, dtheta, phi, dphi, fg, fx)])
    def residuals_shooting(q):
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
        guesstheta, guessphi = q
        shoot = solve_ivp(rhs_shooting, interval, [bcs[0], guesstheta, bcs[2], guessphi])
        return shoot.y[2, -1] - bcs[1], shoot.y[3, -1] - bcs[0]

    sol = scipy.optimize.root(residuals_shooting, [0,0])
    guesstheta, guessphi = sol.x
    slists = np.linspace(interval[0], interval[1], N+1)
    sol_shoot = solve_ivp(rhs_shooting, interval, [bcs[0], guesstheta, bcs[2], guessphi], t_eval=slists)
    return slists, sol_shoot.y    


#The function used to draw on all hairs for the problem
def drawing_hair(L, R, fx, fg, theta0):
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
    
    assert L == abs(L), "L must be an positive number"
    assert L != 0, "L must be an positive number"
    assert R == abs(R), "R must be an positive number"
    assert R != 0, "R must be an positive number"
    assert fx == float(fx), "fx must be a float"
    assert fg == float(fg), "fg must be a float"
    
    #Initialise the list of hairs
    hair1 = []
    #Initialise the loop for theta0
    for j in theta0:
        #Returning the list of s values and shooting solution
        hairs, hair = shooting_dirichlet(equation3a, [0,L], [j,0], 100, fg, fx)
        #Defining both x0 and z0 for each j in theta0
        x0 = R*np.cos(j)
        z0 = R*np.sin(j)
        def right(s, z):
            x,z,theta,dtheta = z
            return np.array([dx(theta), dz(theta), dtheta, equation3a(s, theta, dtheta, fg, fx)])
        #Solving for the x and z coordinates
        hair_xandz=solve_ivp(right, [0,L],[x0, z0, hair[0][0], hair[1][0]], t_eval=hairs)
        #Defining for x and z coordinates
        xcoord, zcoord = hair_xandz.y[0], hair_xandz.y[1]
        #Adding the x and z coordinates to an array
        hair1.append([xcoord, zcoord])
    return hair1

#Same function as above and trying to change it to a 3D version
def drawing_hair1(L, R, fx, fg, theta0, phi0):
    """
       ---Inputs---
       L - Float, length of the hair
       R - Float, radius of the spherical head
       fx - Float, force from wind acting on each hair in the positive x direction
       fg - Float, force from gravity acting on each hair in the negative z direction
       theta0 - List, specifies the list of values where hair meets head
       phi0 - 
       
       ---Returns---
       (x,z) - Coordinates, of the hairs on the head
    """
    
    assert L == abs(L), "L must be an positive number"
    assert L != 0, "L must be an positive number"
    assert R == abs(R), "R must be an positive number"
    assert R != 0, "R must be an positive number"
    assert fx == float(fx), "fx must be a float"
    assert fg == float(fg), "fg must be a float"
    
    hair1 = []
    for j in theta0:
        for k in phi0:
            hairs, hair = shooting_dirichlet1(d2theta, d2phi, [0,L], [j,0,k,0], 100, fg, fx)
            x0 = R*np.cos(j)*np.cos(k)
            y0 = -R*np.cos(j)*np.sin(k)
            z0 = R*np.sin(j)
            def right(s, z):
                x,y,z,theta,dtheta,phi,dphi = z
                return np.array([dx1(theta, phi), dy(theta, phi), dz1(theta), dtheta, d2theta(s, theta, dtheta, phi, dphi, fg, fx) ,dphi, d2phi(s, theta, dtheta, phi, dphi, fg, fx)])
            hair_xandyandz = solve_ivp(right, [0, L], [x0,y0,z0,hair[0][0], hair[1][0], hair[2][0], hair[3][0]])
            xcoord, ycoord, zcoord = hair_xandyandz.y[0], hair_xandyandz.y[1], hair_xandyandz.y[2]
            hair1.append([xcoord, ycoord, zcoord])
    return hair1
             
#Plotting the graph no wind force and only gravity force
def task2(L,R):
    """
       ---Inputs---
       L - Float, length of the hair
       R - Float, radius of the spherical head
       
       ---Returns---
       Graph - Plotting x against z of the hairs with no force due to wind and only gravity
       """
    #Drawing the hairs from 0-pi
    hairs = drawing_hair(L, R, 0, 0.1, np.linspace(0, np.pi, 20))
    #Defining the circle for the hairs to be drawn on
    theta = np.linspace(0, 2*np.pi, 100)
    x1 = R*np.cos(theta)
    x2 = R*np.sin(theta)
    #Plot the circle
    plt.plot(x1,x2)
    #Plotting each individual hair
    for j in hairs:
        plt.plot(j[0], j[1], 'k')
    #Setting limits for  the x and y axis
    plt.xlim(-15,15)
    plt.ylim(-15,15)
    plt.xlabel("x")
    plt.ylabel("z")
    #Set the title
    plt.title("fx = 0, fg = 0.1, Task 2")

#Plotting the graph with wind force and gravity force
def task3(L,R):
    """
       ---Inputs---
       L - Float, length of the hair
       R - Float, radius of the spherical head
       
       ---Returns---
       Graph - Plotting x against z of the hairs with force due to wind and gravity
       """
    hairs = drawing_hair(L, R, 0.1, 0.1, np.linspace(0, np.pi, 20))
    #Defining the circle for the hairs to be drawn on
    theta = np.linspace(0, 2*np.pi, 100)
    x1 = R*np.cos(theta)
    x2 = R*np.sin(theta)
    #Plot the circle
    plt.plot(x1,x2)
    #Plotting each individual hair
    for j in hairs:
        plt.plot(j[0], j[1], 'k')
    #Setting limits for the x and y axis
    plt.xlim(-15,15)
    plt.ylim(-15,15)
    #Set x and y labels
    plt.xlabel("x")
    plt.ylabel("z")
    #Set the title
    plt.title("fx = 0.1, fg = 0.1, Task 3")
    
def task4(L,R):
    hairs = drawing_hair1(L, R, 0.05, 0.1, np.linspace(0, 0.49*np.pi, 10), np.linspace(0, np.pi, 10))
    u = np.linspace(0, np.pi, 100)
    v = np.linspace(0, 2 * np.pi, 100)
    x = np.outer(3*np.sin(u), 3*np.sin(v))
    y = np.outer(3*np.sin(u), 3*np.cos(v))
    z = np.outer(3*np.cos(u), 3*np.ones_like(v))
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z, linewidth=0.0, cstride = 1, rstride = 1)
    for j in hairs:
        plt.plot(j[0], j[1], j[2])
    plt.xlim()
    plt.ylim()
    plt.title("3d plot of hairs on the x,y and z planes for Task 4")
    
    plt.figure()
    for j in hairs:
        plt.plot(j[0], j[2])
    plt.xlabel("x")
    plt.ylabel("z")
    plt.xlim(-15,15)
    plt.ylim(-15,15)
    plt.title("x and z plot for Task 4")
    
    plt.figure()
    for j in hairs:
        plt.plot(j[1], j[2])
    plt.xlabel("y")
    plt.ylabel("z")
    plt.xlim(-15,15)
    plt.ylim(-15,15)
    plt.title("y and z plot for Task 4")






