# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:28:51 2019

@author: Isaac
"""


from matplotlib import pyplot
import numpy

#Coursework 1
#Task 1

def bvector(x):
    return numpy.array([0,0])

#def bvector2(x):
    #return numpy.array([[numpy.cos(10*x)-10*numpy.sin(10*x)],[199*numpy.cos(10*x)-10*numpy.sin(10*x)],[208*numpy.cos(10*x)+10000*numpy.sin(10*x)]])
      

#RK3(numpy.array([[-1000, 0], [1000, -1]]), bvector, numpy.array([1,0]), [0, 0.1], 10)
#DIRK3(numpy.array([[-1000, 0], [1000, -1]]), bvector, numpy.array([1,0]), [0, 0.1], 10)
         
def RK3(A, bvector, y0, interval, N):
    """A - nxn matrix
       bvector - n vector of initial data 
       y0 - n vector of actual intial data 
       interval - list of values 
       N - integer value of how many times to iterate"""
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
    for x in xpoints[1:]:
        #use yn to calculate y1 and y2
        y1 = yn + h * (numpy.dot(A, yn) + bvector(x))
        y2 = (3/4) * yn + (1/4) * y1 + (1/4) * h * (numpy.dot(A, y1) + (bvector(x + h)))
        #updating yn
        yn = (1/3) * yn + (2/3) * y2 + (2/3) * h * (numpy.dot(A, y2) + (bvector(x + h)))
        #adding yn to the empty numpy array at the end of the loop
        z = numpy.append(z, yn)
        
    #reshaping the vector into a matrix as required
    z = z.reshape((-1,2))
    #changing the shape of the matrix so that it satisfies the n x (N+1) condition
    z = z.transpose()

    #pyplot.plot(xpoints[:], z[1,:])
    #pyplot.plot(xpoints[:], z[0,:])
    #pyplot.xlabel("xpoints")
    #pyplot.ylabel("z") 
    #pyplot.title("xpoints compared with z")
    #pyplot.show()
    
    return xpoints, z
    

def DIRK3(A, bvector, y0, interval, N):
       """A - nxn matrix
       bvector - n vector of initial data 
       y0 - n vector of actual intial data 
       interval - list of values 
       N - integer value of how many times to iterate"""
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
       g = 1.5 / (3 + 3**0.5)
       l = 1.5 * (1 + 3**0.5) / (3 + 3**0.5)
       for x in xpoints[1:]:
           A0 = numpy.identity(numpy.shape(A)[0]) - h*u*A
           b0 = yn + h * u * bvector(x + h*u)
           y1 = numpy.linalg.solve(A0, b0)
           
           A1 = numpy.identity(numpy.shape(A)[0]) - h*u*A
           b1 = y1 + h * v * (numpy.dot(A, y1) + bvector(x + h*u)) + h * u * bvector(x + h*v + 2*h*u)
           y2 = numpy.linalg.solve(A1, b1)
           
           yn = (1 - l) * yn + l * y2 + h * g * (numpy.dot(A, y2) + bvector(x + h*v + 2*h*u))
           z = numpy.append(z, yn)
       #reshape vector again
       z = z.reshape((-1,2))
       #change shape as required
       z = z.transpose()
       
       #pyplot.plot(xpoints[:], z[1,:])
       #pyplot.plot(xpoints[:], z[0,:])
       #pyplot.xlabel("$Insert x label here$")
       #pyplot.ylabel("$\log(x) \sin(2 \pi x)$") 
       #pyplot.title("Insert title here")
       #pyplot.show()
       
       return xpoints, z
    
           
def task3(A, bvector, y0, interval):
    U = []
    B = []
    C = []
    points = range(1, 11)
    errors=[]
    for k in points:
        h = (interval[1] - interval[0]) / 40*(k)
        U = numpy.append(U, h)
        X, Y = RK3(A, bvector, y0, interval, 40*(k))
        V, W = DIRK3(A, bvector, y0, interval, 40*(k))
        s = 0 
        t = 0
        ysol = 1000/999 * (numpy.exp(-X[1:]) - numpy.exp(-1000*X[1:]))
        errors.append(h*numpy.sum(numpy.abs((Y[1:, 1]-ysol)/ysol)))
        for l in range(1, 40*(k)+1):
            y2e = (1000/999) * (numpy.exp(-X[l]) - numpy.exp(-1000*X[l]))
            s = s + h * abs((Y[1, l] - y2e) / y2e)
            t = t + h * abs((W[1, l] - y2e) / y2e)
        B = numpy.append(B, s)
        C = numpy.append(C, t)
        
    f = numpy.poly1d(numpy.polyfit(U, B, 3))
    g = numpy.poly1d(numpy.polyfit(U, C, 3))
    
    print(U)
    print(B)
    
    pyplot.loglog(U, errors, label="RK3")
    pyplot.loglog(U[::-1], C, label="DIRK3")
    pyplot.loglog(U, f(U), label="Polyfit for RK3")
    pyplot.loglog(U, g(U), label="Polyfit for DIRK3")
    pyplot.legend()
    pyplot.show()
        
          
    return print("Error of RK3 is:"), print(B), print("Error for DIRK3 is"), print(C)


def task4(bvector, y0, interval):
    A = numpy.array([[-1, 0, 0], [-99, -100, 0], [-10098, 9900, -10000]])
    U = []
    B = []
    C = []
    points = range(4, 17)
    for k in points:
        h = (interval[1] - interval[0]) / 200*(k)
        U = numpy.append(U, h)
        X, Y = RK3(A, bvector, y0, interval, 200*(k))
        V, W = DIRK3(A, bvector, y0, interval, 200*(k))
        s = 0 
        t = 0
        for l in range(2, 200*(k)+1):
            y2e = (1000/999) * (numpy.exp(-X[l]) - numpy.exp(-1000*X[l]))
            s = s + h * abs((Y[1, l] - y2e) / y2e)
            t = t + h * abs((W[1, l] - y2e) / y2e)
        B = numpy.append(B, s)
        C = numpy.append(C, t)




        

        
    
           
           
       
       
    






