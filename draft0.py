""" define the quadratic pairs """
from sympy import Symbol
import numpy as np
from sympy.physics.wigner import wigner_3j
import matplotlib.pyplot as plt
import cwignerd
import camb


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


class Call_Wignerd:
    def __init__(self, npoints):
        self.npoints = npoints
        self.zvec, self.wvec = cwignerd.init_gauss_legendre_quadrature(npoints)

    def cf_from_cl(self, s1, s2, cl):

        lmax = len(cl)-1
        return cwignerd.wignerd_cf_from_cl(s1, s2, 1, self.npoints, lmax, self.zvec, cl)

# I need to define the cl which is the factor before the wigner-d function defined in the (46)(47)(48). cl should include the lensed information given by the camp


#Value = Call_Wignerd(100)
#s1 = 2
#s2 = 2
#cl = np.linspace(0, 1, 20)
#print(Value.cf_from_cl(s1, s2, cl))
#plt.plot(Value.zvec, Value.cf_from_cl(s1, s2, cl))
# plt.show()


class Zeta_factor:
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2

    def lensed_cl(self):
        """ should return different functions of lensed_cl depending on s1 and s2 """
        return np.linspace(0, 1, 100)

    def factor_of_zeta(self):
        if (self.s1 == 0) & (self.s2 == 0):
            return 1
        if (self.s1 == 0) & (self.s2 == 1):
            return 2
        if (self.s1 == 0) & (self.s2 == -1):
            return 3
        if (self.s1 == 1) & (self.s2 == 1):
            return 4
        if (self.s1 == 1) & (self.s2 == -1):
            return 5


Factor = Zeta_factor(0, 0)
print(Factor.factor_of_zeta())


class Zeta:

    def __init__(self, s1, s2, npoints):
        self.s1 = s1
        self.s2 = s2
        self.npoints = npoints
        self.zvec, self.wvec = cwignerd.init_gauss_legendre_quadrature(npoints)

    def lensed_cl(self):
        """ should return different functions of lensed_cl depending on s1 and s2 """
        return np.linspace(0, 1, 100)

    def unlensed_cl(self):
        return np.linspace(0, 1, 100)

    def zeta_factor(self):
        if (self.s1 == 0) & (self.s2 == 0):
            return self.lensed_cl()
        if (self.s1 == 0) & (self.s2 == 1):
            return self.lensed_cl()
        if (self.s1 == 0) & (self.s2 == -1):
            return self.lensed_cl()
        if (self.s1 == 1) & (self.s2 == 1):
            return self.lensed_cl()
        if (self.s1 == 1) & (self.s2 == -1):
            return self.lensed_cl()

    def zeta_function(self):

        lmax = len(self.zeta_factor())-1
        return cwignerd.wignerd_cf_from_cl(self.s1, self.s2, 1, self.npoints, lmax, self.zvec, self.zeta_factor())


Test1 = Zeta(1, 1, 100)
Test2 = Zeta(0, 1, 100)

plt.plot(Test1.zvec, Test1.zeta_factor())
plt.plot(Test2.zvec, Test2.zeta_factor())
plt.show()
