from healpy import sphtfunc, visufunc
import numpy as np
import matplotlib.pyplot as plt
import path
import load_data
import noise_model
import ipdb
import reconstruction_noise
import read_map
import importlib

importlib.reload(read_map)


# reconstruction
class rec_TT:
    """ using TT estimator to reconstruct phi_lm and phi_cl using healpy """
    def __init__(self, *args):
        """input spectra"""
        self.lmax = args[0]
        self.nside = args[1]
        self.Cl_unlsd = args[2][0:self.lmax]

        self.lm_obs = args[3]
        self.Cl_obs = args[4][0:self.lmax]
        self.norm = args[5][0:self.lmax]

    def factor(self):
        ell = np.arange(0, self.lmax)
        return ell * (ell + 1)

    def weight1(self):
        ell = np.arange(0, self.lmax)

        lm_1 = sphtfunc.almxfl(self.lm_obs, (-ell * (ell + 1) / self.Cl_obs))
        map1 = sphtfunc.alm2map(lm_1, self.nside)

        lm_2 = sphtfunc.almxfl(self.lm_obs, self.Cl_unlsd / self.Cl_obs)
        map2 = sphtfunc.alm2map(lm_2, self.nside)

        lm_ret = sphtfunc.map2alm(map1 * map2, lmax=self.lmax)
        return lm_ret

    def weight2(self):
        ell = np.arange(0, self.lmax)

        lm_1 = sphtfunc.almxfl(self.lm_obs, 1 / self.Cl_obs)
        map1 = sphtfunc.alm2map(lm_1, self.nside)

        lm_2 = sphtfunc.almxfl(self.lm_obs,
                               (ell * (ell + 1) * self.Cl_unlsd / self.Cl_obs))
        map2 = sphtfunc.alm2map(lm_2, self.nside)

        lm_ret = sphtfunc.map2alm(map1 * map2, lmax=self.lmax)
        return lm_ret

    def weight3(self):
        ell = np.arange(0, self.lmax)

        lm_1 = sphtfunc.almxfl(self.lm_obs, 1 / self.Cl_obs)
        map1 = sphtfunc.alm2map(lm_1, self.nside)

        lm_2 = sphtfunc.almxfl(self.lm_obs, self.Cl_unlsd / self.Cl_obs)
        map2 = sphtfunc.alm2map(lm_2, self.nside)

        lm_ret = sphtfunc.almxfl(sphtfunc.map2alm(map1 * map2, lmax=self.lmax),
                                 self.factor())
        return lm_ret

    def lm_d(self):
        ell = np.arange(0, self.lmax)
        lm_d = sphtfunc.almxfl(
            self.weight1() + self.weight2() + self.weight3(),
            1 / 2 * self.norm[ell] / (np.sqrt(self.factor())))
        return lm_d

    def var_dd(self):
        ell = np.arange(0, self.lmax)
        return sphtfunc.alm2cl(self.lm_d())
