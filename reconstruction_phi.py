import numpy as np
import matplotlib.pyplot as plt
import wignerd
import path
import pandas as pd
import load_data
import numpy
import noise_model
import ipdb

lmax = 3000
lmin = 2
ls = np.arange(0, lmax + 1)
nlev_t = 27.  # telmperature noise level, in uk.arclmin
nlev_p = np.sqrt(2) * 40
bealm_fwhlm = 7

noise = noise_model.noise(bealm_fwhlm, lmax, nlev_t, nlev_p)
nltt = noise.tt()
nlee = noise.ee()
nlbb = noise.bb()


class phi_TT:
    """reconstruction of phi_lm and phi spectra"""

    def __init__(self, *args):
        """input spectra"""
        self.lmin = args[0]
        self.lmax = args[1]
        self.zeta = wignerd.gauss_legendre_quadrature(4501)
        self.unlensed_TT = load_data.unlensed(self.lmin, self.lmax,
                                              'TT').spectra()
        self.lensed_TT = load_data.lensed(self.lmin, self.lmax,
                                          'TT').spectra() + nltt

    def factor1(self):
        ell = np.arange(self.lmin, self.lmax + 1)
        return ell * (ell + 1)

#    ipdb.set_trace()

    def factor2(self):
        ell = np.arange(self.lmin, self.lmax + 1)
        return (2 * ell + 1)**(1 / 2)

    def part11(self, m, m1):
        cl = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl[ell] = (-1)**ell * (-1 * self.factor1() * self.factor2()
                               ) * self.unlensed_TT[ell] / self.lensed_TT[ell]
        return self.zeta.cf_from_cl(m1, 0, cl)

    def part12(self, m, m1):
        cl = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl[ell] = (-1)**ell * (self.factor2()) / self.lensed_TT[ell]
        return self.zeta.cf_from_cl(-m - m1, 0, cl)

    def block1(self, m, m1):
        ret = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl = self.zeta.cl_from_cf(self.lmax, m, 0,
                                  self.part11(m, m1) * self.part12(m, m1))
        ret[ell] = (1 / 2 * (-1)**ell * cl[ell])
        return ret


#    def part21(self):
# def spectra(self, ell):
#     m = np.arange(-2 * ell, 2 * ell)
#     return np.sum(self.lm(ell, m)**2)

    """ Phi = phi(lmin, lmax)   Phi.lm(ell,m) and Phi.spectra(ell) """

Phi_TT = phi_TT(lmin, lmax)
print(Phi_TT.block1(1, 1)[2:2000])

ls = np.arange(0, lmax + 1)

plt.plot(ls[2:2000], Phi_TT.block1(1, 1)[2:2000])
plt.show()


class zeta:
    """calculate zeta functions"""

    def __int__(self, *args):
        """input the lmin, lmax and spectra"""
        self.lmin = args[0]
        self.lmax = args[1]
        self.estimator = args[2]
        self.quadrature = wignerd.gauss_legendre_quadature(
            3 * self.lmax / 2 + 1)
        self.TT_unlensed = load_data.unlensed(self.lmin, self.lmax,
                                              'TT').spectra()
        self.TT_lensed = load_data.lensed(self.lmin, self.lmax, 'TT').spectra()

    def factors(self):
        """calculate the factor arrays"""

        if self.estimator == 'TT':
            return np.arange(self.lmax)

    def zeta(self, n1, n2):
        """calculate zeta functions, n1 and n2 are the parameters of the wignerd functions in them, cl is the arrays input before the wignerd functions"""
        ell = np.arange(self.lmin, self.lmax + 1)
        return self.zeta.cf_from_cl(n1, n2, self.factor)
