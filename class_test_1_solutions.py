import pytest

import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import scipy
from scipy import integrate
from scipy import optimize

def question_1():
    """
    Solution to question 1 goes here
    """
    A = numpy.array([[1.0, 3.0], [4.0, 5.0]])
    return numpy.linalg.matrix_power(A, 3)

def question_2():
    """
    Solution to question 2 goes here
    """
    v = numpy.linspace(1.0, 2.0, 40)
    return numpy.dot(v, v)
    

def question_3():
    """
    Solution to question 3 goes here
    """
    v = numpy.linspace(1.0, 2.0, 40)
    w = v**2 * numpy.cos(numpy.pi * v)
    return numpy.sum(w[::2])

def question_4():
    """
    Solution to question 4 goes here
    """
    A = [[8.0, 2.0, 4.0], [2.0, -12.0, 6.0], [4.0, 6.0, 1.0]]
    b = [4.0, 9.0, 2.0]
    return numpy.linalg.solve(A, b)

def question_5():
    """
    Solution to question 5 goes here
    """
    A = [[8.0, 2.0, 4.0], [2.0, -12.0, 6.0], [4.0, 6.0, 1.0]]
    evals, evecs = numpy.linalg.eig(A - 2.0 * numpy.eye(3))
    return numpy.min(numpy.abs(evals)), numpy.max(numpy.abs(evals))


def question_6():
    """
    Solution to question 6 goes here
    """
    A = [[8.0, 2.0, 4.0], [2.0, -12.0, 6.0], [4.0, 6.0, 1.0]]
    return numpy.tril(A,-1)

def question_7():
    """
    Solution to question 7 goes here
    """
    x = numpy.arange(1.0, 4.01, 0.01)
    y = numpy.log(x)*numpy.sin(2.0*numpy.pi*x)
    pyplot.plot(x, y)
    pyplot.xlabel(r"$x$")
    pyplot.ylabel(r"$\log(x) \sin(2 \pi x)$")
    pyplot.title("Figure for question 7")
    pyplot.show()

def question_8():
    """
    Solution to question 8 goes here
    """
    x = numpy.arange(0.0, 2.05, 0.05)
    y = numpy.arange(0.0, 1.05, 0.05)
    X, Y = numpy.meshgrid(x, y)
    Z = numpy.exp(-X**2)*numpy.cos(2.0*numpy.pi*Y**2)
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap = cm.hot)
    pyplot.show()

def question_9():
    """
    Solution to question 9 goes here
    """
    def f(s):
        return numpy.cos(s)**3 - s - s**2
    root = optimize.newton(f, 1.0)
    return root


def question_10():
    """
    Solution to question 10 goes here
    """
    y0 = [0.25, 0.75]
    y = numpy.zeros((50, 2))
    y[0, :] = y0
    for i in range(49):
        u, v = y[i, :]
        y[i+1, :] = [numpy.cos(v), numpy.sin(u)]
    return y[4, :], y[-1, :]


if __name__ == "__main__":
    pytest.main()
    question_7()
    question_8()
    