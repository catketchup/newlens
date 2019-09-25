import numpy as np
import matplotlib.pyplot as plt
import wignerd
import path
import pandas as pd
import scipy.interpolate
import load_data
import noise_model
# calculate the EB estimator using lmy new result as a delmon
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


class TT:
    def __init__(self, *args):
        self.lmin = args[0]
        self.lmax = args[1]
        self.zeta = wignerd.gauss_legendre_quadrature(4501)
        self.array1 = load_data.unlensed(self.lmin, self.lmax, 'TT').spectra()
        self.array2 = load_data.lensed(self.lmin, self.lmax,
                                       'TT').spectra() + nltt

    def zeta_00(self):
        cl_00 = np.zeros(self.lmax + 1, dtype=complex)
        ell = np.arange(self.lmin, self.lmax + 1)
        cl_00[ell] = (2 * ell + 1) / (4 * np.pi) * (1. / self.array2[ell])
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
            self.zeta_00() * self.zeta_11() - self.zeta_01() * self.zeta_01()
        ) + self.zeta.cl_from_cf(self.lmax, 1, -1,
                                 self.zeta_00() * self.zeta_1n1() -
                                 self.zeta_01() * self.zeta_0n1())

        ell = np.arange(self.lmin, self.lmax + 1)

        ret[ell] = (np.pi * clL[ell])**-1
        return ret


class EB:
    def __init__(self, *args):
        self.lmin = args[0]
        self.lmax = args[1]
        self.zeta = wignerd.gauss_legendre_quadrature(4501)
        self.array1 = load_data.unlensed(self.lmin, self.lmax, 'EE').spectra()
        self.array2 = load_data.lensed(self.lmin, self.lmax,
                                       'EE').spectra() + nlbb
        self.array3 = load_data.lensed(self.lmin, self.lmax,
                                       'BB').spectra() + nlbb


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


def t(ell):
    return (ell * (ell + 1.))


if 0:
    TT_nl = t(ls) * t(ls) * TT(lmin, lmax).noise() / (2 * np.pi)
    TT_ql = np.real(np.loadtxt("data/TT.dat", dtype=complex))
    plt.plot(ls[2:2000], TT_nl[2:2000])
    plt.plot(ls[2:2000], TT_ql[2:2000])
    plt.legend(['TT_ql', 'TT_nl'])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$L$')
    plt.ylabel(r'$[L[L+1] C_L^{\phi\phi} / 2\pi$')
    plt.show()
    print('done')

if 1:
    filenalme_TT = './data/hu_TT.csv'
    filenalme_EB = './data/hu_EB.csv'

    TT_hu = np.array(pd.read_csv(filenalme_TT))
    EB_hu = np.array(pd.read_csv(filenalme_EB))
    TT_nl = t(ls) * TT(lmin, lmax).noise() / (2 * np.pi)
    EB_nl = t(ls) * EB(lmin, lmax).noise() / (2 * np.pi)

# colmpared with Hu's data given by Mat
if 1:
    plt.plot(ls[2:2000], TT_nl[2:2000])
    plt.plot(TT_hu[:, 0], TT_hu[:, 1])
    plt.legend(['TT_noise', 'TT_noise_hu'])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$L$')
    plt.ylabel(r'$[L[L+1] C_L^{\phi\phi} / 2\pi$')
    plt.show()

if 1:
    plt.plot(ls[2:2000], EB_nl[2:2000])
    plt.plot(EB_hu[:, 0], EB_hu[:, 1])
    plt.legend(['EB_noise', 'EB_noise_hu'])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$L$')
    plt.ylabel(r'$[L[L+1] C_L^{\phi\phi} / 2\pi$')
    plt.show()

if 0:
    f = scipy.interpolate.interp1d(TT_hu[:, 0], TT_hu[:, 1])
    xnew = np.arange(3, 1500)
    ynew = f(xnew)
    print(ynew)
