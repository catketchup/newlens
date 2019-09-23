import numpy as np
import matplotlib.pyplot as plt
import wignerd
import path
import pandas as pd
import load_data
import numpy
import noise_model

lmax = 3000
lmin = 2
ls = np.arange(0, lmax + 1)
nlev_t = 27.  # telmperature noise level, in uk.arclmin
nlev_p = np.sqrt(2) * 40
bealm_fwhlm = 7


class phi:
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

    def factor2(self):
        ell = np.arange(self.lmin, self.lmax + 1)
        return (2 * ell + 1)**(1 / 2)

    def lm(self, ell, m):
        ell = np.arange(self.lmin, self.lmax + 1)
        cl1 = np.zeros(self.lmax + 1, dtype=complex)
        cl2 = np.zeros(self.lmax + 1, dtype=complex)

        cl1[ell] = (
            -1)**ell * self.unlensed_TT / self.lensed_TT * self.factor1()
        return ell * m

    def spectra(self, ell):
        m = np.arange(-2 * ell, 2 * ell)
        return np.sum(self.lm(ell, m)**2)

    """ Phi = phi(lmin, lmax)   Phi.lm(ell,m) and Phi.spectra(ell) """


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
