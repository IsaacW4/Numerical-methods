# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:28:51 2019

@author: Isaac
"""


from matplotlib import pyplot
import numpy

#Coursework 1


#bvector to be used within RK3, DIRK3 and Task 3
def b(x):
    return numpy.array([0,0])

#bvector to be used within Task 4
def b2(x):
    return numpy.array([numpy.cos(10*x) - 10*numpy.sin(10*x), 199*numpy.cos(10*x) - 10*numpy.sin(10*x), 208*numpy.cos(10*x) + 10000*numpy.sin(10*x)])
      
#Both of these were used as easy ways to copy and paste the code during testing of Task 3
#RK3(numpy.array([[-1000, 0], [1000, -1]]), bvector, numpy.array([1,0]), [0, 0.1], 10)
#DIRK3(numpy.array([[-1000, 0], [1000, -1]]), bvector, numpy.array([1,0]), [0, 0.1], 10)
         

#All of these are functions that are used in the error analysis for Task 3 and 4 to compare our approximations too
def y1exact(x):
    return numpy.exp(-1000*x)

def y2exact(x):
    return (1000/999)*(numpy.exp(-x) - numpy.exp(-1000*x))

def y3exact(x):
    return numpy.cos(10*x) - numpy.exp(-x)

def y4exact(x):
    return numpy.cos(10*x) + numpy.exp(-x) - numpy.exp(-100*x)

def y5exact(x):
    return numpy.sin(10*x) + 2*numpy.exp(-x) - numpy.exp(-100*x) - numpy.exp(-10000*x)
  
    
    
    
def RK3(A, bvector, y0, interval, N):
    """
    Computes RK3 algorithm
    
       ---Inputs---
    
       A - nxn matrix
       bvector - n vector of initial data 
       y0 - n vector of actual intial data 
       interval - list of values 
       N - integer value of how many times to iterate
       
       ---Returns---
       
       xpoints - The locations of y denoted x_j of size N+1 
       z - The approximation of y at location x_j of size n x N+1
       
       """
       
       
    #Asserts for RK3 to check the input
    assert N == abs(int(N)), "N has to be a positive interger"
    assert len(interval) == 2, "Interval should only have 2 numbers"
    assert interval[1] > interval[0], "Interval second element must be larger than the first"
    assert N != 0, "There cannot be 0 iterations"
    assert numpy.shape(A)[1] == numpy.shape(A)[0], "A should be of shape n x n"
    assert numpy.shape(A)[0] == numpy.shape(bvector(0))[0], "bvector should be an n array"
    assert numpy.shape(A)[0] == numpy.shape(y0)[0], "y0 should be an n array"


    #Both of these values of A and y0 were used during the test stage of the code, and was used to ensure I was getting suitable results
    #A = numpy.array([[-100, 0], [100, -1]])
    #y0 = numpy.array([1,0])
    #intialising the array to be adding values to within the loop
    z = numpy.array([y0])
    #getting first value of interval
    a = interval[0]
    #getting last value of interval
    b = interval[-1]
    #defining step size
    h = (b - a) / N 
    #selecting the range of points to iterate over
    xpoints = numpy.linspace(a, b, num = N+1)
    #Set starting point
    yn = y0
    #iterating to find y_n+1
    for x in xpoints[:-1]:
        #use yn to calculate y1 and y2
        y1 = yn + h * (numpy.dot(A, yn) + bvector(x))
        y2 = (3/4) * yn + (1/4) * y1 + (1/4) * h * (numpy.dot(A, y1) + bvector(x + h))
        #updating yn
        yn = (1/3) * yn + (2/3) * y2 + (2/3) * h * (numpy.dot(A, y2) + bvector(x + h))
        #adding yn to the empty numpy array at the end of the loop
        z = numpy.append(z, yn)
    #reshaping the vector into a matrix as required
    z = z.reshape((-1,numpy.shape(y0)[0]))
    #changing the shape of the matrix so that it satisfies the n x (N+1) condition
    z = z.transpose()
    
    return xpoints, z
    

def DIRK3(A, bvector, y0, interval, N):
       """
    Computes DIRK 3 algorithm 
       
    
       ---Inputs---
       
       A - nxn matrix
       bvector - n vector of initial data 
       y0 - n vector of actual intial data 
       interval - list of values 
       N - integer value of how many times to iterate
       
       ---Returns---
       
       xpoints - The locations of y denoted x_j of size N+1 
       z - The approximation of y at location x_j of size n x N+1
       
       """
       
       #Asserts for RK3 to check the input
       assert N == abs(int(N)), "N has to be a positive interger"
       assert len(interval) == 2, "Interval should only have 2 numbers"
       assert interval[1] > interval[0], "Interval second element must be larger than the first"
       assert N != 0, "There cannot be 0 iterations"
       assert numpy.shape(A)[1] == numpy.shape(A)[0], "A should be of shape n x n"
       assert numpy.shape(A)[0] == numpy.shape(bvector(0))[0], "bvector should be an n array"
       assert numpy.shape(A)[0] == numpy.shape(y0)[0], "y0 should be an n array"
       
       
       #More parts of code used during testing
       #A = numpy.array([[-100, 0], [100, -1]])
       #y0 = numpy.array([1,0])
       a = interval[0]
       b = interval[-1]
       h = (b - a) / N 
       z = numpy.array([y0])
       #selecting the range of points to iterate over
       xpoints = numpy.linspace(a, b, num = N+1)
       #Set starting point
       yn = y0
       #define all variables necessary for y1 and y2
       u = 0.5*(1 - 3**(-0.5))
       v = 0.5*(3**0.5 - 1)
       g = 3 / (2*(3 + 3**0.5))
       l = (1.5 * (1 + 3**0.5)) / (3 + 3**0.5)
       for x in xpoints[:-1]:
           #constructing an identity matrix of shape required
           A0 = numpy.identity(numpy.shape(A)[0]) - h*u*A
           b0 = yn + h * u * bvector(x + h*u)
           #using a black box method to solve Ax=b
           y1 = numpy.linalg.solve(A0, b0)
           
           A1 = numpy.identity(numpy.shape(A)[0]) - h*u*A
           b1 = y1 + h * v * (numpy.dot(A, y1) + bvector(x + h*u)) + h * u * bvector(x + h*v + 2*h*u)
           y2 = numpy.linalg.solve(A1, b1)
           
           #computing yn / updating yn
           yn = (1 - l) * yn + l * y2 + h * g * (numpy.dot(A, y2) + bvector(x + h*v + 2*h*u))
           #adding yn to empty numpy array
           z = numpy.append(z, yn)
       #reshape vector again
       z = z.reshape((-1,numpy.shape(y0)[0]))
       #change shape as required
       z = z.transpose()  
       
       return xpoints, z
    
           
def task3(A, bvector, y0, interval):
    """
 Plots the graphs of the errors of RK3 and DIRK3
 
     ---Inputs---
     
     A - nxn matrix
     bvector - n vector of initial data 
     y0 - n vector of actual intial data 
     interval - list of values
     
    ---Returns---
    
    rk3error - summation of h * error of yn approximate compared with yn exact of the rk3 algorithm
    dirk3 error - summation of h * error of yn approximate compared with yn exact of the dirk3 algorithm
    Figure 1 - RK3 error with polyfit
    Figure 2 - DIRK3 error with polyfit
    Figure 3:
        - RK3 y1 compared with y1 exact
        - RK3 y2 compared with y2 exact
    Figure 4:
        - DIRK3 y1 compared with y1 exact
        - DIRK3 y1 compared with y1 exact
    
    """
    
    
    #defining all quantities necessary for the error comparison between the exact values and approximations
    hpoints = []
    rk3error = []
    dirk3error = []
    points = range(1, 11)
    for k in points:
        h = (interval[1] - interval[0]) / (40*(k))
        #adding all points in the range to hpoints which are essentially step points
        hpoints = numpy.append(hpoints, h)
        #running the approximations for rk3 and dirk3
        X, Y = RK3(A, bvector, y0, interval, 40*(k))
        V, W = DIRK3(A, bvector, y0, interval, 40*(k))
        #setting quantities to 0
        s = 0 
        t = 0
        for l in range(1, 40*(k)+1):
            #computing the exact value for y2
            y2e = (1000/999) * (numpy.exp(-X[l]) - numpy.exp(-1000*X[l]))
            #error for rk3
            s = s + h * abs((Y[1, l] - y2e) / y2e)
            #eror for dirk3
            t = t + h * abs((W[1, l] - y2e) / y2e)
        #summing up the erros from both rk3 and dirk3
        rk3error = numpy.append(rk3error, s)
        dirk3error = numpy.append(dirk3error, t)
        
   
    #using polyfit for a line of best fit of both errors
    rk3co = numpy.polyfit(numpy.log(hpoints[1:10]), numpy.log(rk3error[1:10]), 1)
    dirk3co = numpy.polyfit(numpy.log(hpoints), numpy.log(dirk3error), 1)
    
    #Figure 1
    #plotting a graph of the 1 norm for y2 of rk3
    pyplot.figure()
    pyplot.plot(numpy.log(hpoints), numpy.log(rk3error), 'rx', label="rk3")
    #using a log of hpoints and the coefficients from the polyfit, and "%.5f" % to limit the amount of decimal places the gradient is given to
    pyplot.plot(numpy.log(hpoints), (rk3co[0]*(numpy.log(hpoints))+rk3co[1]), label="Polyfit for RK3: " +  str("%.5f" % rk3co[0]))
    pyplot.xlabel("h - step length")
    pyplot.ylabel("1 norm of error of y2 for rk3")
    pyplot.title("1 norm for the error of y2 in rk3 against step lengths h ")
    pyplot.legend()
    
    #Figure 2
    #plotting a graph of the 1 norm for y2 of dirk3
    pyplot.figure()
    pyplot.plot(numpy.log(hpoints), numpy.log(dirk3error), 'rx', label="dirk3")
    #using a log of hpoints and the coefficients from the polyfit, and "%.5f" % to limit the amount of decimal places the gradient is given to
    pyplot.plot(numpy.log(hpoints), (dirk3co[0]*(numpy.log(hpoints))+dirk3co[1]), label="Polyfit for DIRK3: " + str("%.5f" % dirk3co[0]))
    pyplot.xlabel("h - step length")
    pyplot.ylabel("1 norm of error of y2 for dirk3")
    pyplot.title("1 norm for the error of y2 in dirk3 against step lengths h")
    pyplot.legend()
    
    #Figure 3
    #giving y1 and y2 approximations for RK3 against the xpoints
    fig, (ax1, ax2) = pyplot.subplots(1,2, figsize = (12,4))
    ax1.semilogy(X, Y[0,:], 'kx', label="y1 approximation")
    ax1.semilogy(X, y1exact(X), 'c', label="y1")
    ax1.set_xlabel("x")
    ax1.set_ylabel("$log(y)$")
    ax1.set_title("y1 rk3 approximations against xpoints")
    ax1.legend()
    ax2.plot(X, Y[1,:], 'kx', label="y2 approximation")
    ax2.plot(X, y2exact(X), 'c', label="y2")
    ax2.set_xlabel("x")
    ax2.set_ylabel("$y$")
    ax2.set_title("y2 rk3 approximations against xpoints")
    ax2.legend()

    #Figure 4
    #giving y1 and y2 approximations for DIRK3 against the xpoints
    fig, (ax3, ax4) = pyplot.subplots(1,2, figsize = (12,4))
    ax3.semilogy(V, W[0,:], 'kx', label="y1 approximation")
    ax3.semilogy(X, y1exact(X), 'c', label="y1")
    ax3.set_xlabel("x")
    ax3.set_ylabel("$log(y)$")
    ax3.set_title("y1 dirk3 approximations against xpoints")
    ax3.legend()
    ax4.plot(V, W[1,:], 'kx', label="y2 approximation")
    ax4.plot(X, y2exact(X), 'c', label="y2")
    ax4.set_xlabel("x")
    ax4.set_ylabel("$y$")
    ax4.set_title("y2 dirk3 approximations against xpoints")
    ax4.legend()
          
    #Giving both the erros for 10 points of RK3 and DIRK3
    return print("Error of RK3 is: "), print(rk3error), print("Error for DIRK3 is: "), print(dirk3error)


def task4(bvector, y0, interval):
    """
  Plots the graphs of the errors of rk3 and dirk3 for a specific case specified by the coursework

    ---Inputs---
    
     bvector - n vector of initial data 
     y0 - n vector of actual intial data 
     interval - list of values
     
     ---Returns---
     
    rk3error - summation of h * error of yn approximate compared with yn exact of the rk3 algorithm
    dirk3 error - summation of h * error of yn approximate compared with yn exact of the dirk3 algorithm
    Figure 5 - DIRK3 error with polyfit
    Figure 6:
        - RK3 y1 compared with y1 exact
        - RK3 y2 compared with y2 exact
        - RK3 y3 compared with y3 exact
    Figure 7:
        - DIRK3 y1 compared with y1 exact
        - DIRK3 y2 compared with y2 exact
        - DIRK3 y3 compared with y3 exact
    
    """
    #defining all quantities and also defining A in this specific case
    A = numpy.array([[-1, 0, 0], [-99, -100, 0], [-10098, 9900, -10000]])
    hpoints = []
    rk3error = []
    dirk3error = []
    #using a different range from Task 3
    points = range(4, 17)
    for k in points:
        #mainly the same code has been recycled from Task 3 to Task 4 and instead using the 200K for 40K (A much bigger quantity is needed for convergence to be shown)
        h = (interval[1] - interval[0]) / (200*(k))
        hpoints = numpy.append(hpoints, h)
        X, Y = RK3(A, bvector, y0, interval, 200*(k))
        V, W = DIRK3(A, bvector, y0, interval, 200*(k))
        s = 0 
        t = 0
        for l in range(1, 200*(k)+1):
            y2e = (numpy.sin(10*X[l]) + 2*numpy.exp(-X[l]) - numpy.exp(-100*X[l]) - numpy.exp(-10000*X[l]))
            s = s + h * abs((Y[2, l] - y2e) / y2e)
            t = t + h * abs((W[2, l] - y2e) / y2e)
        rk3error = numpy.append(rk3error, s)
        dirk3error = numpy.append(dirk3error, t)
    
    
    
    dirk3co = numpy.polyfit(numpy.log(hpoints[2:,]), numpy.log(dirk3error[2:,]), 1)
    print(dirk3error)
    print(hpoints)
    #Figure 5
    #Showing the error of y2 against step lengths h 
    pyplot.figure()
    pyplot.plot(numpy.log(hpoints), numpy.log(dirk3error), 'rx', label="dirk3")
    #using a log of hpoints and the coefficients from the polyfit, and "%.5f" % to limit the amount of decimal places the gradient is given to
    pyplot.plot(numpy.log(hpoints), (dirk3co[0]*(numpy.log(hpoints))+dirk3co[1]), label="Polyfit for DIRK3: " + str("%.5f" % dirk3co[0]))
    pyplot.xlabel("h - step length")
    pyplot.ylabel("1 norm of error of y2 for dirk3")
    pyplot.title("1 norm for the error of y2 in dirk3 against step lengths h ")
    pyplot.legend()
    
    
    #Figure 6
    #plotting 3 graphs of y1, y2 and y3 of rk3 approximations against the xpoints, showing the convergence of rk3 dropping off
    fig, (ax1, ax2, ax3) = pyplot.subplots(1,3, figsize = (16,4))
    ax1.plot(X, Y[0,:], 'kx', label="y1 approximation")
    ax1.plot(X, y3exact(X), 'c', label="y1")
    ax1.set_xlabel("x")
    ax1.set_ylabel("$y1$")
    ax1.set_title("y1 rk3 approximations against xpoints")
    ax1.legend()
    ax2.plot(X, Y[1,:], 'kx', label="y2 approximation")
    ax2.plot(X, y4exact(X), 'c', label="y2")
    ax2.set_xlabel("x")
    ax2.set_ylabel("$y2$")
    ax2.set_title("y2 rk3 approximations against xpoints")
    ax2.legend()
    ax3.plot(X, Y[2,:], 'kx', label="y3 approximation")
    ax3.plot(X, y5exact(X), 'c', label="y3")
    ax3.set_xlabel("x")
    ax3.set_ylabel("$y3$")
    ax3.set_title("y3 rk3 approximations against xpoints")
    ax3.legend()
    
    #Figure 7 
    #again, plotting 3 graphs of y1, y2 and y3 of dirk3 approximations against the xpoints, showing the dirk3 convergence is consistent
    fig, (ax4, ax5, ax6) = pyplot.subplots(1,3, figsize = (16,4))
    ax4.plot(V, W[0,:], 'kx', label="y1 approximation")
    ax4.plot(X, y3exact(X), 'c', label="y1")
    ax4.set_xlabel("x")
    ax4.set_ylabel("$y1$")
    ax4.set_title("y1 dirk3 approximations against xpoints")
    ax4.legend()
    ax5.plot(V, W[1,:], 'kx', label="y2 approximation")
    ax5.plot(X, y4exact(X), 'c', label="y2")
    ax5.set_xlabel("x")
    ax5.set_ylabel("$y2$")
    ax5.set_title("y2 dirk3 approximations against xpoints")
    ax5.legend()
    ax6.plot(V, W[2,:], 'kx', label="y3 approximation")
    ax6.plot(X, y5exact(X), 'c', label="y3")
    ax6.set_xlabel("x")
    ax6.set_ylabel("$y3$")
    ax6.set_title("y3 dirk3 approximations against xpoints")
    ax6.legend()
    
    
RK3(numpy.array([[-1000, 0], [1000, -1]]), b, numpy.array([1,0]), [0, 0.1], 10)
DIRK3(numpy.array([[-1000, 0], [1000, -1]]), b, numpy.array([1,0]), [0, 0.1], 10)
task3(numpy.array([[-1000, 0], [1000, -1]]), b, numpy.array([1,0]), [0,0.1])
task4(b2, numpy.array([0,1,0]), [0,1])

