import numpy as np
import matplotlib.pyplot as plt
from newlens import math, load_data, noise_model
import path
import pandas as pd
import scipy.interpolate
import importlib

importlib.reload(load_data)


class TT:
    def __init__(self, *args):
        self.lmin = args[0]
        self.lmax = args[1]
        self.bealm_fwhlm = args[2]
        self.nlev_t = args[3]  # telmperature noise level, in uk.arclmin
        self.nlev_p = args[4]
        self.nltt = noise_model.noise(self.bealm_fwhlm, self.lmax, self.nlev_t,
                                      self.nlev_p).tt()

        self.zeta = math.wignerd.gauss_legendre_quadrature(4501)
        self.array1 = load_data.unlensed(self.lmin, self.lmax, 'TT').spectra()
        self.array2 = load_data.lensed(self.lmin, self.lmax,
                                       'TT').spectra() + self.nltt

    def zeta_00(self):
        cl_00 = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl_00[ell] = (2 * ell + 1) / (4 * np.pi) * (1 / self.array2[ell])
        return self.zeta.cf_from_cl(0, 0, cl_00)

    def zeta_01(self):
        cl_01 = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl_01[ell] = (2*ell+1)/(4*np.pi)*np.sqrt(ell*(ell+1)) * \
                     (self.array1[ell]/self.array2[ell])
        return self.zeta.cf_from_cl(0, 1, cl_01)

    def zeta_0n1(self):
        cl_0n1 = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl_0n1[ell] = (2*ell+1)/(4*np.pi)*np.sqrt(ell*(ell+1)) * \
                      (self.array1[ell] / self.array2[ell])
        return self.zeta.cf_from_cl(0, -1, cl_0n1)

    def zeta_11(self):
        cl_11 = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl_11[ell] = (2 * ell+1)/(4*np.pi)*ell*(ell+1) * \
                     (self.array1[ell]**2)/(self.array2[ell])
        return self.zeta.cf_from_cl(1, 1, cl_11)

    def zeta_1n1(self):
        cl_1n1 = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl_1n1[ell] = (2*ell+1)/(4*np.pi)*ell*(ell+1) * \
            self.array1[ell]**2/(self.array2[ell])
        return self.zeta.cf_from_cl(1, -1, cl_1n1)

    def noise(self):
        ret = np.zeros(self.lmax + 1, dtype=complex)
        clL = self.zeta.cl_from_cf(
            self.lmax, -1, -1,
            self.zeta_00() * self.zeta_11() -
            self.zeta_01() * self.zeta_01()) + self.zeta.cl_from_cf(
                self.lmax, 1, -1,
                self.zeta_00() * self.zeta_1n1() -
                self.zeta_01() * self.zeta_0n1())

        ell = np.arange(self.lmin, self.lmax + 1)

        ret[ell] = (np.pi * clL[ell])**-1
        return ret


class EB:
    def __init__(self, *args):
        self.lmin = args[0]
        self.lmax = args[1]
        self.bealm_fwhlm = args[2]
        self.nlev_t = args[3]  # telmperature noise level, in uk.arclmin
        self.nlev_p = args[4]
        self.nlee = noise_model.noise(self.bealm_fwhlm, self.lmax, self.nlev_t,
                                      self.nlev_p).ee()
        self.nlbb = noise_model.noise(self.bealm_fwhlm, self.lmax, self.nlev_t,
                                      self.nlev_p).bb()

        self.zeta = wignerd.gauss_legendre_quadrature(4501)

        self.array1 = load_data.unlensed(self.lmin, self.lmax, 'EE').spectra()
        self.array2 = load_data.lensed(self.lmin, self.lmax,
                                       'EE').spectra() + self.nlee
        self.array3 = load_data.lensed(self.lmin, self.lmax,
                                       'BB').spectra() + self.nlbb


#    import ipdb
#    ipdb.set_trace()

    def zeta_33(self):
        cl_33 = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl_33[ell] = (2 * ell + 1) / (4 * np.pi) * (
            self.array1[ell])**2 / self.array2[ell] * (ell - 2) * (ell + 3)

        return self.zeta.cf_from_cl(3, 3, cl_33)

    def zeta_3n3(self):
        cl_3n3 = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl_3n3[ell] = (2 * ell + 1) / (4 * np.pi) * (
            self.array1[ell])**2 / self.array2[ell] * (ell - 2) * (ell + 3)

        return self.zeta.cf_from_cl(3, -3, cl_3n3)

    def zeta_31(self):
        cl_31 = np.zeros(self.lmax + 1, dtype=np.complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl_31[ell] = (2*ell+1)/(4*np.pi) * \
            (self.array1[ell])**2/self.array2[ell] * \
            np.sqrt((ell-1)*(ell+2)*(ell-2)*(ell+3))

        return self.zeta.cf_from_cl(3, 1, cl_31)

    def zeta_3n1(self):
        cl_3n1 = np.zeros(self.lmax + 1, dtype=np.complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl_3n1[ell] = (2*ell+1)/(4*np.pi) * \
            (self.array1[ell])**2/self.array2[ell] * \
            np.sqrt((ell-1)*(ell+2)*(ell-2)*(ell+3))

        return self.zeta.cf_from_cl(3, -1, cl_3n1)

    def zeta_11(self):
        cl_11 = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl_11[ell] = (2*ell+1)/(4*np.pi) * self.array1[ell]**2 / \
            self.array2[ell]*(ell-1)*(ell+2)

        return self.zeta.cf_from_cl(1, 1, cl_11)

    def zeta_1n1(self):
        cl_1n1 = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl_1n1[ell] = (2*ell+1)/(4*np.pi) * self.array1[ell]**2 / \
            self.array2[ell]*(ell-1)*(ell+2)

        return self.zeta.cf_from_cl(1, -1, cl_1n1)

    def zeta_22(self):
        cl_22 = np.zeros(self.lmax + 1)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl_22[ell] = (2 * ell + 1) / (4 * np.pi) * (1. / self.array3[ell])
        return self.zeta.cf_from_cl(2, 2, cl_22)

    def zeta_2n2(self):
        cl_22 = np.zeros(self.lmax + 1)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl_22[ell] = (2 * ell + 1) / (4 * np.pi) * (1. / self.array3[ell])
        return self.zeta.cf_from_cl(2, -2, cl_22)

    def noise(self):
        ret = np.zeros(self.lmax + 1, dtype=np.complex)
        clL = self.zeta.cl_from_cf(
            self.lmax, 1, 1,
            self.zeta_33() * self.zeta_22() -
            2 * self.zeta_3n1() * self.zeta_2n2() +
            self.zeta_11() * self.zeta_22()) - self.zeta.cl_from_cf(
                self.lmax, 1, -1,
                self.zeta_3n3() * self.zeta_2n2() - 2 * self.zeta_31() *
                self.zeta_22() + self.zeta_1n1() * self.zeta_2n2())
        ell = np.arange(self.lmin, self.lmax + 1)

        ret[ell] = ((np.pi / 4.) * clL[ell])**-1
        return ret
