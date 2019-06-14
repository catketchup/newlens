""" define the quadratic pairs """
from sympy import Symbol
import numpy as np
from sympy.physics.wigner import wigner_3j


class Norm:
    """ define functions for calculating tt nomarlization """

    def __init__(self, *args):
        self.ell1 = args[0]
        self.ell = args[1]
        self.ell2 = args[2]
        self.spin = args[3]
        self.C_l1_TT_unlens = args[4]
        self.C_l2_TT_unlens = args[5]

    def H_func(self):
        """ define the coefficient of F """

        ell1 = self.ell1
        ell = self.ell
        ell2 = self.ell2
        return (-1*ell1*(ell1+1)+ell*(ell+1)+ell2*(ell2+1)) * \
            np.sqrt((2*ell1+1)*(2*ell+1)*(2*ell2+1)/(16*np.pi))

    def F_func_2(self):
        """ define the F in eq(14) with the l2, L, l1 order """
        ell1 = self.ell1
        ell = self.ell
        ell2 = self.ell2
        spin = self.spin
        return self.H_func()*wigner_3j(ell2, ell, ell1, spin, 0, -spin)

    def F_func_1(self):
        """ define the F in eq(14) with the l1, L, l2 order """

        ell1 = self.ell1
        ell = self.ell
        ell2 = self.ell2
        spin = self.spin
        return self.H_func()*wigner_3j(ell1, ell, ell2, spin, 0, -spin)

    def f_TT(self):
        """ define the f quadratic pairs of TT """
        return self.C_l1_TT_unlens*self.F_func_2()+self.C_l2_TT_unlens*self.F_func_1()


TT_NORM = Norm(2, 4, 6, 0, 1, 1)
print(TT_NORM.H_func())
print(TT_NORM.F_func_1())
print(TT_NORM.f_TT())
