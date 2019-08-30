""" be ready to calculate the tt normalizaion integral """
import numpy as np
import ctypes
from sympy import Symbol


class NormIntegral:
    """
    to calculate the tt normalizaion
    I need to evaluate the wigner d-functions by upward recursion in l

    """

    """
    as a test, I firstly plan to implement the l=1, l=2 terms
    """

    def __init__(self, *args):
        self.ell_max = args[0]
        self.ell = args[1]
        self.ell2 = args[2]
        """ by which form the """
        """ need to consider how to calculate the summation """

    def zeta_00_T(self):
        """ the term with 00 """

        return self.ell_max

    def zeta_01_T(self):
        """ the term with 0+/-1 """

        return self.ell

    def zeta_11_T(self):
        """ the term with 1+/-1 """
        return self.ell2


Test = NormIntegral(1, 2, 3)
print(Test.zeta_00_T())
