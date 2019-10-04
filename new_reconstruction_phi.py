from healpy import sphtfunc
import numpy as np
import matplotlib
import path
import load_data
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
    """ reconstruction of phi_lm and phi spectra using healpy """

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


## alm from a map
in_powspec = np.arange(2, 1000)
healpymap = sphtfunc.synfast(in_powspec, 2)
alm = sphtfunc.map2alm(healpymap)
#out_powspec = sphtfunc.anafast(healpymap)
m = np.array([0, 1])
index = sphtfunc.Alm.getidx(7, 1, m)
#print(index)
#print(alm[index])
#print(out_powspec)
print(alm)

lmin = 2
lmax = 3000


## test healpy.sphtfunc.almxfl
def fl1():
    ell = np.arange(lmin, lmax + 1)
    return ell * (ell + 1)


factor1 = sphtfunc.almxfl(alm, fl1())
print(factor1)

map1 = sphtfunc.alm2map(factor1, 2)
print(map1)


def fl2():
    ell = np.arange(lmin, lmax + 1)
    return (2 * ell + 1)**(1 / 2)


factor2 = sphtfunc.almxfl(alm, fl2())

map2 = sphtfunc.alm2map(factor2, 2)
print(map2)

map = map1 * map2
print(map)

phi_lm = sphtfunc.map2alm(map)
print(phi_lm)
