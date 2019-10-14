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

    # def factor1(self):
    #     ell = np.arange(self.lmin, self.lmax + 1)
    #     return ell * (ell + 1)

    # def factor2(self):
    #     ell = np.arange(self.lmin, self.lmax + 1)
    #     return (2 * ell + 1)**(1 / 2)

    def part1(self):
        ell = np.arange(self.lmin, self.lmax + 1)

        cl1 = np.sqrt(4 * np.pi) * (-1)**(
            ell + 1) * self.unlensed_TT[ell] / self.lensed_TT[ell]
        factor1 = sphtfunc.almxfl(self.alm, cl1)
        map1 = sphtfunc.alm2map(factor1, self.nside)

        cl2 = np.sqrt(4 * np.pi) * (-1)**(ell + 1) * (ell * (
            ell + 1)) / self.lensed_TT[ell]
        factor2 = sphtfunc.almxfl(self.alm, cl2)
        map2 = sphtfunc.alm2map(factor2, self.nside)

        phi_lm = sphtfunc.map2alm(map1 * map2)

        phi_cl = (1 / (np.sqrt(4 * np.pi)) * (-1)**
                  (ell + 1))**2 * sphtfunc.alm2cl(phi_lm)[ell] / (ell *
                                                                  (ell + 1))**2
        # visufunc.mollview(map1 * map2)
        # plt.show()
        #return reconstruction_noise.TT(self.lmin, self.lmax).noise()
        return phi_cl

    def part2(self):
        return 1


nside = 400  #nside should be large enough
test = phi_TT(lmin, lmax, nside)
phi_cl = np.array(test.part1())
lmax = 1000
lmin = 2
bealm_fwhlm = 7
nlev_t = 27.  # telmperature noise level, in uk.arclmin
nlev_p = np.sqrt(2) * 40
noise_tt = reconstruction_noise.TT(lmin, lmax, bealm_fwhlm, nlev_t,
                                   nlev_p).noise()
plt.plot(np.square(noise_tt)[0:999] * phi_cl)
plt.show()
