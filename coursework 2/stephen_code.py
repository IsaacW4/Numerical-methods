# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:03:25 2019

@author: Isaac
"""
L = 4
R = 10


import numpy
from scipy.optimize import root, brentq
from scipy.integrate import solve_ivp, solve_bvp
from matplotlib import pyplot

def Shooting_2D(d2theta, interval,bcs,N,fg,fx):
    """ Solved BVP in 2D using the shooting method"""
    def RHS_Shooting(s,q):
        theta,dtheta=q
        return numpy.array([dtheta,d2theta(s,theta,dtheta,fg,fx)])
    def residuals_shooting(alpha_0):
        shot=solve_ivp(RHS_Shooting,interval,[bcs[0],alpha_0])
        return shot.y[1,-1]-bcs[1]
    alpha_0=brentq(residuals_shooting,-10,10)
    slist=numpy.linspace(interval[0],interval[1],N+2)
    solution=solve_ivp(RHS_Shooting,interval,[bcs[0],alpha_0],t_eval=slist)
    return slist,solution.y


def d2theta(s, theta, dtheta, phi, dphi, fg, fx):
    return s*fg*numpy.cos(theta)

def d2phi(s, theta, dtheta, phi, dphi, fg, fx):
    return s*fx*numpy.cos(phi)*numpy.sin(theta)

def equation3_2D(s,theta,dtheta,fg,fx):
    return s*fg*numpy.cos(theta)+s*fx*numpy.sin(theta)

def dx_2D(theta):
    return numpy.cos(theta)

def dz_2D(theta):
    return numpy.sin(theta)

def GenHair_2D(L,R,fx,fg,Thetas):
    hairs = []
    for i in Thetas:
        hair_s,hair=Shooting_2D(equation3_2D,[0,L],[i,0],100,fg,fx)
        x0=R*numpy.cos(i)
        z0=R*numpy.sin(i)
        def RHS(s,q):
            x,z,theta,dtheta=q
            return numpy.array([dx_2D(theta),dz_2D(theta),dtheta,equation3_2D(s,theta,dtheta,fg,fx)])
        hairxz=solve_ivp(RHS,[0,L],[x0, z0, hair[0][0], hair[1][0]] ,t_eval=hair_s)
        x,z=hairxz.y[0],hairxz.y[1]
        hairs.append([x,z])
    return hairs
    
    
Hair_Grav = GenHair_2D(L, R, 0, 0.1, numpy.linspace(0, numpy.pi, 20))
fig = pyplot.figure()
ax = fig.add_subplot(111)
head = pyplot.Circle([0,0], R, fill = False, color = "b")
for i in Hair_Grav:
    ax.plot(i[0], i[1], color = "k")
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.add_artist(head)


Hair_Wind = GenHair_2D(L, R, 0.1, 0.1, numpy.linspace(0, numpy.pi, 20))
fig2 = pyplot.figure()
ax2 = fig2.add_subplot(111)
head2 = pyplot.Circle([0,0], R, fill = False, color = "b")
for i in Hair_Wind:
    ax2.plot(i[0], i[1], color = "k")
ax2.set_xlim(-15, 15)
ax2.set_ylim(-15, 15)
ax2 .add_artist(head2)




