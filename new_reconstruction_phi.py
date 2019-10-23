from healpy import sphtfunc, visufunc
import numpy as np
import matplotlib.pyplot as plt
import path
import load_data
import noise_model
import ipdb
import reconstruction_noise


class rec_TT:
    """ using TT estimator to reconstruct phi_lm and phi_cl using healpy """

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
        self.alm = sphtfunc.map2alm(self.input_map, lmax=self.lmax)
        self.norm = reconstruction_noise.TT(self.lmin, self.lmax, bealm_fwhlm,
                                            nlev_t, nlev_p).noise()

    def factor_a(self):
        ell = np.arange(self.lmin, self.lmax + 1)
        return ell * (ell + 1)

    def part1(self):
        ell = np.arange(self.lmin, self.lmax + 1)

        cl1 = -self.factor_a() / self.lensed_TT[ell]
        alm1 = sphtfunc.almxfl(self.alm, cl1)
        map1 = sphtfunc.alm2map(alm1, self.nside)

        cl2 = self.unlensed_TT[ell] / self.lensed_TT[ell]
        alm2 = sphtfunc.almxfl(self.alm, cl2)
        map2 = sphtfunc.alm2map(alm2, self.nside)

        ret_lm = sphtfunc.map2alm(map1 * map2, lmax=self.lmax)
        return ret_lm

    def part2(self):
        ell = np.arange(self.lmin, self.lmax + 1)

        cl1 = 1 / self.lensed_TT[ell]
        alm1 = sphtfunc.almxfl(self.alm, cl1)
        map1 = sphtfunc.alm2map(alm1, self.nside)

        cl2 = self.factor_a() * self.unlensed_TT[ell] / self.lensed_TT[ell]
        alm2 = sphtfunc.almxfl(self.alm, cl2)
        map2 = sphtfunc.alm2map(alm2, self.nside)

        ret_lm = sphtfunc.map2alm(map1 * map2, lmax=self.lmax)
        return ret_lm

    def part3(self):
        ell = np.arange(self.lmin, self.lmax + 1)

        cl1 = 1 / self.lensed_TT[ell]
        alm1 = sphtfunc.almxfl(self.alm, cl1)
        map1 = sphtfunc.alm2map(alm1, self.nside)

        cl2 = self.unlensed_TT[ell] / self.lensed_TT[ell]
        alm2 = sphtfunc.almxfl(self.alm, cl2)
        map2 = sphtfunc.alm2map(alm2, self.nside)

        ret_lm = sphtfunc.almxfl(
            sphtfunc.map2alm(map1 * map2, lmax=self.lmax), self.factor_a())
        return ret_lm

    def lm(self):
        ell = np.arange(self.lmin, self.lmax + 1)

        # ret_alm = self.part1()
        ret_alm = sphtfunc.almxfl(self.part1() + self.part2() + self.part3(),
                                  1 / 2 * self.norm[ell] * 1 / self.factor_a())
        return ret_alm

    def cl(self):
        ell = np.arange(self.lmin, self.lmax + 1)
        ret_cl = sphtfunc.alm2cl(self.lm())
        return ret_cl


# class phi_EB():
#     """ using EB estimator to reconstruct phi_lm and phi_cl using healpy """

#     def __init__(self, *args):
#         """input spectra"""
#         self.lmin = args[0]
#         self.lmax = args[1]
#         self.nside = args[2]
#         self.unlensed_EE = load_data.unlensed(self.lmin, self.lmax,
#                                               'EE').spectra()
#         self.lensed_EE = load_data.lensed(self.lmin, self.lmax,
#                                           'EE').spectra() + nlee
#         self.lensed_BB = load_data.lensed(self.min, self.lmax,
#                                           'BB').spectra() + nlbb
#         self.input_map_EE = sphtfunc.synfast(
#             self.lensed_EE, self.nside)  #create a map from self.lensed_TT
#         self.input_map_BB = sphtfunc.synfast(self.lensed_BB, self.nside)
#         self.alm_EE = np.conj(sphtfunc.map2alm(self.input_map_EE))
#         self.alm_BB = np.conj(sphtfunc.map2alm(self.input_map_BB))
#         self.norm = reconstruction_noise.TT(self.lmin, self.lmax, bealm_fwhlm,
#                                             nlev_t, nlev_p).noise()

#     def factor_a(self):
#         ell = np.arange(self.lmin, self.lmax + 1)
#         return ell * (ell + 1)

#     def part11(self):
#         ell = np.arange(self.lmin, self.lmax + 1)

nside = 400
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


def t(ell):
    return (ell * (ell + 1.))


Noise_cl_TT = t(ls) * reconstruction_noise.TT(lmin, lmax, bealm_fwhlm, nlev_t,
                                              nlev_p).noise() / (2 * np.pi)
Input_cldd = t(ls) * load_data.input_cldd(lmin, lmax).spectra() / (2 * np.pi)
Rec_cldd_TT = rec_TT(lmin, lmax, nside).cl()

if 1:  ## plot of phi_cl_TT
    plt.clf()
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(ls[lmin:lmax - 1], Noise_cl_TT[lmin:lmax - 1])
    plt.plot(ls[lmin:lmax - 1], Input_cldd[lmin:lmax - 1])
    plt.plot(ls[lmin:lmax - 1], (Input_cldd + Noise_cl_TT)[lmin:lmax - 1])
    plt.plot(ls[lmin:lmax - 1],
             t(ls)[lmin:lmax - 1] * Rec_cldd_TT[lmin:lmax - 1] / (2 * np.pi))
    plt.legend(['Noise_TT', 'Input_cldd', 'Noise_TT +Input_cldd', 'Rec_cldd'])
    plt.show()
