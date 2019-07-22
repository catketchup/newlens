import numpy as np
import matplotlib.pyplot as plt
import cwignerd
import load_data
import wignerd
# calculate the EB estimator using my new result as a demon
lmax = 3000
ls = np.arange(0, lmax+1)
nlev_t = 5  # temperature noise level, in uk.arcmin
nlev_p = 5
beam_fwhm = 1


# noise spectra
# bl is transfer functions
# nlee = nlbb = (np.pi/180./60.*nlev_p)**2 / bl**2


class unlensed:
    def __init__(self, estimator):
        self.estimator = estimator

    def spectra(self):
        if self.estimator == 'EE':
            return np.loadtxt("planck_lensing_wp_highL_bestFit_20130627_scalCls.dat", usecols=(2), unpack=True)
        if self.estimator == 'BB':
            return np.loadtxt("planck_lensing_wp_highL_bestFit_20130627_scalCls.dat", usecols=(3), unpack=True)


class lensed:
    def __init__(self, estimator):
        self.estimator = estimator

    def spectra(self):
        if self.estimator == 'EE':
            return np.loadtxt("planck_lensing_wp_highL_bestFit_20130627_lensedCls.dat", usecols=(2), unpack=True)
        if self.estimator == 'BB':
            return np.loadtxt("planck_lensing_wp_highL_bestFit_20130627_lensedCls.dat", usecols=(3), unpack=True)


unlensed_ClEE = unlensed('EE').spectra()
lensed_ClEE = lensed('EE').spectra()
unlensed_CLBB = unlensed('BB').spectra()
lensed_ClBB = lensed('BB').spectra()


# how to construct the factors?
# construct Zeta(3,3)
s1 = 3
s2 = 3
glq = wignerd.gauss_legendre_quadrature(4501)
gp1 = glq.cf_from_cl(s1, s2, unlensed_ClEE)
gp2 = glq.cf_from_cl(s1, s2, unlensed_CLBB)
print(glq.zvec)
#plt.hist(glq.zvec, bins=100)
plt.show()
# plt.plot(glq.wvec)
plt.show()
