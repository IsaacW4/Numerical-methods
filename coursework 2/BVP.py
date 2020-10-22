# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:15:53 2019

@author: iw1g17
"""

import numpy
from scipy.optimize import root, root_scalar
from scipy.integrate import solve_ivp, solve_bvp
import matplotlib
from matplotlib import pyplot
matplotlib.rc('font', size=16)


def relax_dirichlet(f, interval, bcs, N):
    """
    Solve a BVP using relaxation (black box root-find)
    
    Parameters
    ----------
    
    f: function
        RHS function defining BVP: y'' = f(x, y, y')
    interval: list of two floats
        [a, b] where BVP defined on x \in [a, b]
    bcs: list of two floats
        Defines Dirichlet boundary values [y(a), y(b)]
    N: single integer
        Number of interior points in the grid
    
    Returns
    -------
    
    x: array of floats
        Location of grid points
    y: array of floats
        Solution at grid points
    """
    x, h = numpy.linspace(interval[0], interval[1], N+2, retstep=True)
    x_interior = x[1:-1]
    y = numpy.zeros_like(x)
    y[0] = bcs[0]
    y[-1] = bcs[-1]
    
    def residuals(y_interior):
        """
        Given guess, compute r = y_guess'' - f(x, y_guess, y_guess')
        
        Parameters
        ----------
        
        y_interior: list of floats
            Values of the guess at the interior grid points
        
        Returns
        -------
        
        residual: list of floats
            Values of residual r at the interior grid points
        """
        residual = numpy.zeros_like(y)
        y_guess = y.copy()
        y_guess[1:-1] = y_interior  # This is the line missed in the lecture
        y_guess[0] = bcs[0]
        y_guess[-1] = bcs[-1]
        for i in range(1, N+1):
            dy = (y_guess[i+1]-y_guess[i-1])/(2*h)
            ddy_h2 = y_guess[i-1] + y_guess[i+1] - 2 * y_guess[i]
            residual[i] = ddy_h2 - h**2 * f(x[i], y_guess[i], dy)
        return residual[1:-1]
    
    sol = root(residuals, numpy.zeros_like(x_interior))
    y[1:-1] = sol.x
    return x, y


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
    x, h = numpy.linspace(interval[0], interval[1], N+2, retstep=True)
    y = numpy.zeros_like(x)
    y[0] = bcs[0]
    
    def rhs_shooting(x, z):
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
        y, dy = z
        return numpy.array([dy, f(x, y, dy)])
    
    def residuals_shooting(dy_a):
        """
        Solve IVP given y'(a): compute r = y(b) - bcs[1]
        
        Parameters
        ----------
        
        dy_a: float
            Guess for derivative of solution at x=a
        
        Returns
        -------
        
        residual: float
            Values of residual r at x=b
        """
        sol_shoot = solve_ivp(rhs_shooting, interval, [bcs[0], dy_a])
        return sol_shoot.y[0, -1] - bcs[1]
    
    sol = root_scalar(residuals_shooting, method='brentq', bracket=[-10, 10])
    dy_a = sol.root
    sol_shoot = solve_ivp(rhs_shooting, interval, [bcs[0], dy_a], t_eval=x)
    y_all = sol_shoot.y
    return x, y_all[0, :]
    

def f(x, y, dy):
    """
    Function defining the BVP
    
    Parameters
    ----------
    
    x: float
        Location
    y: float
        Value of solution
    dy: float
        Value of derivative of solution
        
    Returns
    -------
    
    f: float
        Function defining BVP
    """
    return -1 / (1 + y**2)


def blackbox_f(x, z):
    """
    RHS function for using scipy.integrate.solve_bvp. Matches rhs_shooting.
        
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
    y, dy = z
    return numpy.array([dy, f(x, y, dy)])


interval = [0, 1]
bcs = [0, 0]


def blackbox_bcs(ya, yb):
    """
    Define boundary conditions for using scipy.integrate.solve_bvp.
    """
    return numpy.array([ya[0] - bcs[0],
                        yb[0] - bcs[1]])


print("Relaxation solution")
for N in [5, 50, 500]:
    x, y_relax = relax_dirichlet(f, interval, bcs, N)
    pyplot.plot(x, y_relax, 'bx-', lw=3, mew=3, ms=12)
    pyplot.title(rf"Relaxation, $N={N}$")
    pyplot.show()

print("Shooting solution")
x, y_shoot = shooting_dirichlet(f, interval, bcs, N)
pyplot.plot(x, y_shoot, 'bx-', lw=3, mew=3, ms=12)
pyplot.title("Shooting")
pyplot.show()

print("Black box solution")
y_guess = numpy.zeros((2, len(x)))
sol_bvp = solve_bvp(blackbox_f, blackbox_bcs, x, y_guess)
y_bb = sol_bvp.y[0, :]

print("All solutions")
fig, axes = pyplot.subplots(2, 2, figsize=(12, 12))
axes[0, 0].plot(x, y_relax, 'b-', label='Relaxation')
axes[0, 0].plot(x, y_shoot, 'r:+', label='Shooting')
axes[0, 0].plot(x, y_bb, 'g--x', label='Black box')
axes[0, 0].set_xlabel(r"$x$")
axes[0, 0].set_ylabel(r"$y$")
axes[0, 0].legend()
axes[0, 1].semilogy(x, numpy.abs(y_relax - y_shoot))
axes[0, 1].set_xlabel(r"$x$")
axes[0, 1].set_ylabel(r"$|y_{relax} - y_{shoot}|$")
axes[1, 0].semilogy(x, numpy.abs(y_relax - y_bb))
axes[1, 0].set_xlabel(r"$x$")
axes[1, 0].set_ylabel(r"$|y_{relax} - y_{black box}|$")
axes[1, 1].semilogy(x, numpy.abs(y_bb - y_shoot))
axes[1, 1].set_xlabel(r"$x$")
axes[1, 1].set_ylabel(r"$|y_{black box} - y_{shoot}|$")
fig.tight_layout()