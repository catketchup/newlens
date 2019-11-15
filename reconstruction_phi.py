import numpy as np
import matplotlib.pyplot as plt
import wignerd
import path
import pandas as pd
import load_data
import numpy
import noise_model
import ipdb
import healpy
import pixell

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

    def part11(self, m1):
        cl = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl[ell] = (-1)**ell * (
            -1 * self.factor2()) * self.unlensed_TT[ell] / self.lensed_TT[ell]
        return self.zeta.cf_from_cl(m1, 0, cl)

    def part12(self, m2):
        cl = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl[ell] = (-1)**ell * (
            self.factor1() * self.factor2()) / self.lensed_TT[ell]
        return self.zeta.cf_from_cl(m2, 0, cl)

    def block1(self, m, m1, m2):
        ret = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl = self.zeta.cl_from_cf(self.lmax, m, 0,
                                  self.part11(m1) * self.part12(m2))
        ret[ell] = (1 / 2 * (-1)**ell * self.factor2() * cl[ell])
        return ret

    def part21(self, m1):
        cl = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl[ell] = (-1)**ell * (self.factor1() * self.factor2()
                               ) * self.unlensed_TT[ell] / self.lensed_TT[ell]
        return self.zeta.cf_from_cl(m1, 0, cl)

    def part22(self, m2):
        cl = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl[ell] = (-1)**ell * (self.factor2()) / self.lensed_TT[ell]
        return self.zeta.cf_from_cl(m2, 0, cl)

    def block2(self, m, m1, m2):
        ret = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl = self.zeta.cl_from_cf(self.lmax, m, 0,
                                  self.part21(m1) * self.part22(m2))
        ret[ell] = (1 / 2 * (-1)**ell * self.factor2() * cl[ell])
        return ret

    def part31(self, m1):
        cl = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl[ell] = (-1)**ell * (
            self.factor2()) * self.unlensed_TT[ell] / self.lensed_TT[ell]
        return self.zeta.cf_from_cl(m1, 0, cl)

    def part32(self, m2):
        cl = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl[ell] = (-1)**ell * (self.factor2()) / self.lensed_TT[ell]
        return self.zeta.cf_from_cl(m2, 0, cl)

    def block3(self, m1, m2, m):
        ret = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl = self.zeta.cl_from_cf(self.lmax, m, 0,
                                  self.part31(m1) * self.part32(m2))
        ret[ell] = (1 / 2 *
                    (-1)**ell * self.factor1() * self.factor2() * cl[ell])
        return ret

    def part41(self, m1):
        cl = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl[ell] = (-self.factor1() * self.factor2()) / self.lensed_TT[ell]
        return self.zeta.cf_from_cl(m1, 0, cl)

    def part42(self, m2):
        cl = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl[ell] = self.factor2() * self.unlensed_TT[ell] / self.lensed_TT[ell]
        return self.zeta.cf_from_cl(m2, 0, cl)

    def block4(self, m, m1, m2):
        ret = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl = self.zeta.cl_from_cf(self.lmax, m, 0,
                                  self.part41(m1) * self.part42(m2))
        ret[ell] = 1 / 2 * self.factor2() * cl[ell]
        return ret

    def part51(self, m1):
        cl = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl[ell] = (self.factor2()) / self.lensed_TT[ell]
        return self.zeta.cf_from_cl(m1, 0, cl)

    def part52(self, m2):
        cl = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl[ell] = (self.factor1() * self.factor2()
                   ) * self.unlensed_TT[ell] / self.lensed_TT[ell]
        return self.zeta.cf_from_cl(m2, 0, cl)

    def block5(self, m, m1, m2):
        ret = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl = self.zeta.cl_from_cf(self.lmax, m, 0,
                                  self.part51(m1) * self.part52(m2))
        ret[ell] = (1 / 2 * self.factor2() * cl[ell])
        return ret

    def part61(self, m1):
        cl = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl[ell] = (self.factor1() * self.factor2()) / self.lensed_TT[ell]
        return self.zeta.cf_from_cl(m1, 0, cl)

    def part62(self, m2):
        cl = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl[ell] = (
            self.factor2()) * self.unlensed_TT[ell] / self.lensed_TT[ell]
        return self.zeta.cf_from_cl(m2, 0, cl)

    def block6(self, m, m1, m2):
        ret = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl = self.zeta.cl_from_cf(self.lmax, m, 0,
                                  self.part61(m1) * self.part62(m2))
        ret[ell] = (1 / 2 * self.factor2() * cl[ell])
        return ret


#    def part21(self):
# def spectra(self, ell):
#     m = np.arange(-2 * ell, 2 * ell)
#     return np.sum(self.lm(ell, m)**2)

    """ Phi = phi(lmin, lmax)   Phi.lm(ell,m) and Phi.spectra(ell) """

Phi_TT = phi_TT(lmin, lmax)
ls = np.arange(0, lmax + 1)


def t(ls):
    return ls * (ls + 1)


m1 = 1
m2 = -2
m = 1

# plt.plot(ls[2:2000],
#          Phi_TT.block1(m1, m2, m)[2:2000] + Phi_TT.block2(m1, m2, m)[2:2000] +
#          Phi_TT.block3(m1, m2, m)[2:2000])

#plt.yscale('log')
plt.show()


def plot_seperate(m1, m2, m):
    plt.plot(ls[2:2000], Phi_TT.block1(m1, m2, m)[2:2000])
    plt.plot(ls[2:2000], Phi_TT.block2(m1, m2, m)[2:2000])
    plt.plot(ls[2:2000], Phi_TT.block3(m1, m2, m)[2:2000])
    plt.plot(ls[2:2000], Phi_TT.block4(m1, m2, m)[2:2000])
    plt.plot(ls[2:2000], Phi_TT.block5(m1, m2, m)[2:2000])
    plt.plot(ls[2:2000], Phi_TT.block6(m1, m2, m)[2:2000])
    plt.show()


if 1:
    plot_seperate(m1, m2, m)
