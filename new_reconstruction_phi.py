from healpy import sphtfunc, visufunc
import numpy as np
import matplotlib.pyplot as plt
import path
import load_data
import noise_model
import ipdb
import reconstruction_noise

lmax = 1000
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
    """ reconstruction of phi_lm and phi spectra using healpy """

    def __init__(self, *args):
        """input spectra"""
        self.lmin = args[0]
        self.lmax = args[1]
        self.nside = args[2]
        self.unlensed_TT = load_data.unlensed(self.lmin, self.lmax,
                                              'TT').spectra()
        self.lensed_TT = load_data.lensed(self.lmin, self.lmax,
                                          'TT').spectra() + nltt
        self.input_map = sphtfunc.synfast(
            self.lensed_TT, self.nside)  #create a map from self.lensed_TT
        self.alm = sphtfunc.map2alm(self.input_map)
        self.norm = reconstruction_noise.TT(self.lmin, self.lmax, bealm_fwhlm,
                                            nlev_t, nlev_p).noise()

    def factor_a(self):
        ell = np.arange(self.lmin, self.lmax + 1)
        return ell * (ell + 1)

    def factor_b(self):
        ell = np.arange(self.lmin, self.lmax + 1)
        return (2 * ell + 1)**(1 / 2)

    def part11(self):
        ell = np.arange(self.lmin, self.lmax + 1)

        cl1 = (-1)**ell * self.unlensed_TT[ell] / self.lensed_TT[ell]
        alm1 = sphtfunc.almxfl(self.alm, cl1)
        map1 = sphtfunc.alm2map(alm1, self.nside)

        cl2 = (-1)**ell * self.factor_a() / self.lensed_TT[ell]
        alm2 = sphtfunc.almxfl(self.alm, cl2)
        map2 = sphtfunc.alm2map(alm2, self.nside)

        phi_lm = sphtfunc.almxfl(
            sphtfunc.map2alm(map1 * map2),
            -1 / 2 * (-1)**ell * self.norm[ell] / np.sqrt(self.factor_a()))
        phi_cl = sphtfunc.alm2cl(phi_lm)
        return phi_cl

    def part12(self):
        ell = np.arange(self.lmin, self.lmax + 1)

        cl1 = (-1)**ell * self.factor_a() * self.unlensed_TT[
            ell] / self.lensed_TT[ell]
        alm1 = sphtfunc.almxfl(self.alm, cl1)
        map1 = sphtfunc.alm2map(alm1, self.nside)

        cl2 = (-1)**ell / self.lensed_TT[ell]
        alm2 = sphtfunc.almxfl(self.alm, cl2)
        map2 = sphtfunc.alm2map(alm2, self.nside)

        phi_lm = sphtfunc.almxfl(
            sphtfunc.map2alm(map1 * map2),
            1 / 2 * (-1)**ell * self.norm[ell] / np.sqrt(self.factor_a()))

        phi_cl = sphtfunc.alm2cl(phi_lm)
        return phi_cl

    def part13(self):
        ell = np.arange(self.lmin, self.lmax + 1)

        cl1 = (-1)**ell * self.unlensed_TT[ell] / self.lensed_TT[ell]
        alm1 = sphtfunc.almxfl(self.alm, cl1)
        map1 = sphtfunc.alm2map(alm1, self.nside)

        cl2 = (-1)**ell / self.lensed_TT[ell]
        alm2 = sphtfunc.almxfl(self.alm, cl2)
        map2 = sphtfunc.alm2map(alm2, self.nside)

        phi_lm = sphtfunc.almxfl(
            sphtfunc.map2alm(map1 * map2),
            1 / 2 * (-1)**ell * self.norm[ell] * np.sqrt(self.factor_a()))

        phi_cl = sphtfunc.alm2cl(phi_lm)
        return phi_cl

    def part21(self):
        ell = np.arange(self.lmin, self.lmax + 1)

        cl1 = self.factor_a() / self.lensed_TT[ell]
        alm1 = sphtfunc.almxfl(self.alm, cl1)
        map1 = sphtfunc.alm2map(alm1, self.nside)

        cl2 = self.unlensed_TT[ell] / self.lensed_TT[ell]
        alm2 = sphtfunc.almxfl(self.alm, cl2)
        map2 = sphtfunc.alm2map(alm2, self.nside)

        phi_lm = sphtfunc.almxfl(
            sphtfunc.map2alm(map1 * map2),
            -1 / 2 * self.norm[ell] / np.sqrt(self.factor_a()))
        phi_cl = sphtfunc.alm2cl(phi_lm)
        return phi_cl

    def part22(self):
        ell = np.arange(self.lmin, self.lmax + 1)

        cl1 = 1 / self.lensed_TT[ell]
        alm1 = sphtfunc.almxfl(self.alm, cl1)
        map1 = sphtfunc.alm2map(alm1, self.nside)

        cl2 = self.factor_a() * self.unlensed_TT[ell] / self.lensed_TT[ell]
        alm2 = sphtfunc.almxfl(self.alm, cl2)
        map2 = sphtfunc.alm2map(alm2, self.nside)

        phi_lm = sphtfunc.almxfl(
            sphtfunc.map2alm(map1 * map2),
            1 / 2 * self.norm[ell] / np.sqrt(self.factor_a()))

        phi_cl = sphtfunc.alm2cl(phi_lm)
        return phi_cl

    def part23(self):
        ell = np.arange(self.lmin, self.lmax + 1)

        cl1 = 1 / self.lensed_TT[ell]
        alm1 = sphtfunc.almxfl(self.alm, cl1)
        map1 = sphtfunc.alm2map(alm1, self.nside)

        cl2 = self.unlensed_TT[ell] / self.lensed_TT[ell]
        alm2 = sphtfunc.almxfl(self.alm, cl2)
        map2 = sphtfunc.alm2map(alm2, self.nside)

        phi_lm = sphtfunc.almxfl(
            sphtfunc.map2alm(map1 * map2),
            1 / 2 * self.norm[ell] * np.sqrt(self.factor_a()))

        phi_cl = sphtfunc.alm2cl(phi_lm)
        return phi_cl

    def phi_cl(self):
        return self.part11() + self.part12() + self.part13() + self.part21(
        ) + self.part22() + self.part23()


def t(ell):
    return (ell * (ell + 1.))


nside = 400  #nside should be large enough
test = phi_TT(lmin, lmax, nside)
Phi_cl = np.array(test.phi_cl())
lmax = 1000
lmin = 2
bealm_fwhlm = 7
nlev_t = 27.  # telmperature noise level, in uk.arclmin
nlev_p = np.sqrt(2) * 40
plt.xscale('log')
plt.yscale('log')
plt.plot(Phi_cl / (2 * np.pi))
plt.show()
