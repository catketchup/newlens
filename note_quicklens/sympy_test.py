# https://github.com/sympy/sympy/wiki/Tutorial
from sympy import I
from sympy import *
import numpy as np
x = Symbol('x')
y = Symbol('y')
print('\n')
# symbols
print(x+y)
print(x*y)
print(((x+y)**2).expand())
# differentiation
print(diff(sin(x), x))
# series expansion
print(cos(x).series(x, 0, 10))
# integration
print(integrate(6*x**5, x))  # definite
print(integrate(x**1, (x, -1, 1)))  # indefinite
print(integrate(exp(-x), (x, 0, oo)))  # improper
# complex numbers
print(exp(I*x).expand())
print(exp(I*x).expand(complex=True))
x = Symbol("x", real=True)
print(exp(I*x).expand(complex=True))


def f(x):
    return exp(2*I*x).expand(complex=True)


a_list = ['a', 'b', 'c']
a_list.append('d')
print(f(x))
print(f(45))
print(a_alist)
# printing
print_latex(f(x))
