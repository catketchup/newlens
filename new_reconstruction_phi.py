from healpy import sphtfunc, visufunc
import numpy as np
import matplotlib.pyplot as plt
import path
import load_data
import noise_model
import ipdb
import reconstruction_noise


class phi_TT:
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
        self.alm = np.conj(sphtfunc.map2alm(self.input_map))
        self.norm = reconstruction_noise.TT(self.lmin, self.lmax, bealm_fwhlm,
                                            nlev_t, nlev_p).noise()

    def factor_a(self):
        ell = np.arange(self.lmin, self.lmax + 1)
        return ell * (ell + 1)

    # def factor_b(self):
    #     ell = np.arange(self.lmin, self.lmax + 1)
    #     return (2 * ell + 1)**(1 / 2)

    def part11(self):
        ell = np.arange(self.lmin, self.lmax + 1)

        cl1 = (-1)**ell * self.unlensed_TT[ell] / self.lensed_TT[ell]
        alm1 = sphtfunc.almxfl(self.alm, cl1)
        map1 = sphtfunc.alm2map(alm1, self.nside)

        cl2 = (-1)**ell * self.factor_a() / self.lensed_TT[ell]
        alm2 = sphtfunc.almxfl(self.alm, cl2)
        map2 = sphtfunc.alm2map(alm2, self.nside)

        # ret_lm = sphtfunc.almxfl(
        #     sphtfunc.map2alm(map1 * map2),
        #     1 / 2 * (-1)**ell * self.norm[ell] / self.factor_a())
        ret_lm = -1 / 2 * sphtfunc.map2alm(map1 * map2)
        return ret_lm

    def part12(self):
        ell = np.arange(self.lmin, self.lmax + 1)

        cl1 = (-1)**ell * self.factor_a() * self.unlensed_TT[
            ell] / self.lensed_TT[ell]
        alm1 = sphtfunc.almxfl(self.alm, cl1)
        map1 = sphtfunc.alm2map(alm1, self.nside)

        cl2 = (-1)**ell / self.lensed_TT[ell]
        alm2 = sphtfunc.almxfl(self.alm, cl2)
        map2 = sphtfunc.alm2map(alm2, self.nside)

        ret_lm = 1 / 2 * sphtfunc.map2alm(map1 * map2)
        return ret_lm

    def part13(self):
        ell = np.arange(self.lmin, self.lmax + 1)

        cl1 = (-1)**ell * self.unlensed_TT[ell] / self.lensed_TT[ell]
        alm1 = sphtfunc.almxfl(self.alm, cl1)
        map1 = sphtfunc.alm2map(alm1, self.nside)

        cl2 = (-1)**ell / self.lensed_TT[ell]
        alm2 = sphtfunc.almxfl(self.alm, cl2)
        map2 = sphtfunc.alm2map(alm2, self.nside)

        ret_lm = 1 / 2 * sphtfunc.almxfl(
            sphtfunc.map2alm(map1 * map2), self.factor_a())
        return ret_lm

    def part21(self):
        ell = np.arange(self.lmin, self.lmax + 1)

        cl1 = self.factor_a() / self.lensed_TT[ell]
        alm1 = sphtfunc.almxfl(self.alm, cl1)
        map1 = sphtfunc.alm2map(alm1, self.nside)

        cl2 = self.unlensed_TT[ell] / self.lensed_TT[ell]
        alm2 = sphtfunc.almxfl(self.alm, cl2)
        map2 = sphtfunc.alm2map(alm2, self.nside)

        ret_lm = -1 / 2 * sphtfunc.map2alm(map1 * map2)
        return ret_lm

    def part22(self):
        ell = np.arange(self.lmin, self.lmax + 1)

        cl1 = 1 / self.lensed_TT[ell]
        alm1 = sphtfunc.almxfl(self.alm, cl1)
        map1 = sphtfunc.alm2map(alm1, self.nside)

        cl2 = self.factor_a() * self.unlensed_TT[ell] / self.lensed_TT[ell]
        alm2 = sphtfunc.almxfl(self.alm, cl2)
        map2 = sphtfunc.alm2map(alm2, self.nside)

        ret_lm = 1 / 2 * sphtfunc.map2alm(map1 * map2)
        return ret_lm

    def part23(self):
        ell = np.arange(self.lmin, self.lmax + 1)

        cl1 = 1 / self.lensed_TT[ell]
        alm1 = sphtfunc.almxfl(self.alm, cl1)
        map1 = sphtfunc.alm2map(alm1, self.nside)

        cl2 = self.unlensed_TT[ell] / self.lensed_TT[ell]
        alm2 = sphtfunc.almxfl(self.alm, cl2)
        map2 = sphtfunc.alm2map(alm2, self.nside)

        ret_lm = 1 / 2 * sphtfunc.almxfl(
            sphtfunc.map2alm(map1 * map2), self.factor_a())
        return ret_lm

    def phi_lm(self):
        ell = np.arange(self.lmin, self.lmax + 1)
        alm = self.part11() - self.part12() + self.part13() - self.part21(
        ) + self.part22() + self.part23()
        return sphtfunc.almxfl(alm, self.norm[ell] * np.sqrt(self.factor_a()))

    def phi_cl(self):
        ret_cl = sphtfunc.alm2cl(self.phi_lm())
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

nide = 400
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


Phi_TT = phi_TT(lmin, lmax, nside)
Phi_TT_lm = Phi_TT.phi_lm()
Phi_TT_cl = Phi_TT.phi_cl()
if 0:
    print(Phi_TT_lm)
    Phi_map = sphtfunc.alm2map(Phi_TT_lm, nside)
    visufunc.mollview(Phi_map)

if 1:
    print(np.size(Phi_TT_cl))
if 0:  ## plot of phi_cl_TT

    plt.xscale('log')
    plt.yscale('log')
    plt.plot(ls[lmin:lmax - 1],
             t(ls)[lmin:lmax - 1] * Phi_TT_cl[lmin:lmax - 1])
    # plt.plot(Phi_TT.part11())
    # plt.plot(Phi_TT.part12())
    # plt.plot(Phi_TT.part13())
    # plt.plot(Phi_TT.part21())
    # plt.plot(Phi_TT.part22())
    # plt.plot(Phi_TT.part23())
    # plt.plot(ls[2:1000], (t(ls)[2:2000] * Phi_TT_cl[2:1000]) / (2 * np.pi))
    plt.plot(ls[2:1000], (t(ls) * reconstruction_noise.TT(
        lmin, lmax, bealm_fwhlm, nlev_t, nlev_p).noise())[2:1000] /
             (2 * np.pi))
    plt.plot(ls[2:1000], (t(ls) * reconstruction_noise.EB(
        lmin, lmax, bealm_fwhlm, nlev_t, nlev_p).noise())[2:1000] /
             (2 * np.pi))
    plt.show()
