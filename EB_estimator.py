import numpy as np
import matplotlib.pyplot as plt
import cwignerd
import wignerd
import ipdb

# calculate the EB estimator using my new result as a demon
lmax = 3000
ls = np.arange(0, lmax+1)
nlev_t = 5  # temperature noise level, in uk.arcmin
nlev_p = 5
beam_fwhm = 1


def bl(fwhm_arcmin, lmax):
    """ returns the map-level transfer function for a symmetric Gaussian beam.
         * fwhm_arcmin      = beam full-width-at-half-maximum (fwhm) in arcmin.
         * lmax             = maximum multipole.
    """
    ls = np.arange(0, lmax+1)
    return np.exp(-(fwhm_arcmin * np.pi/180./60.)**2 / (16.*np.log(2.)) * ls*(ls+1.))


bl = bl(beam_fwhm, lmax)

# noise spectra
# bl is transfer functions
nltt = (np.pi/180./60.*nlev_t)**2 / bl**2
nlee = nlbb = (np.pi/180./60.*nlev_p)**2 / bl**2


class unlensed:
    def __init__(self, estimator):
        self.estimator = estimator

    def spectra(self):
        if self.estimator == 'TT':
            array = np.loadtxt(
                'planck_lensing_wp_highL_bestFit_20130627_scalCls.dat', unpack=True)
            return np.concatenate([np.zeros(lmin), array[0: (lmax-lmin), 1]])
        if self.estimator == 'EE':
            return np.loadtxt('planck_lensing_wp_highL_bestFit_20130627_scalCls.dat', unpack=True)
        if self.estimator == 'BB':
            return np.loadtxt('planck_lensing_wp_highL_bestFit_20130627_scalCls.dat', unpack=True)


lmin = 2


class lensed:
    def __init__(self, estimator):
        self.estimator = estimator

    def spectra(self):
        if self.estimator == 'TT':
            return np.loadtxt(
                'planck_lensing_wp_highL_bestFit_20130627_lensedCls.dat', unpack=True)

        if self.estimator == 'EE':
            return np.loadtxt('planck_lensing_wp_highL_bestFit_20130627_lensedCls.dat', unpack=True)
        if self.estimator == 'BB':
            return np.loadtxt('planck_lensing_wp_highL_bestFit_20130627_lensedCls.dat', unpack=True)


# how to construct the factors?
# construct Zeta(0,0)


class TT:
    def __init__(self):
        self.zeta = wignerd.gauss_legendre_quadrature(4501)

    def zeta_00(self):
        cl_00 = np.zeros(lmax+1)
        for ell in range(0, lmax+1):
            cl_00 = (2*ell+1)/(4*np.pi)*(1./(lensed('TT').spectra()+nltt))
        return self.zeta.cf_from_cl(0, 0, cl_00)

    def zeta_01(self):
        cl_01 = np.zeros(lmax+1)
        for ell in range(0, lmax+1):
            cl_01 = (2*ell+1)/(4*np.pi)*np.sqrt(ell*(ell+1)) * \
                (unlensed('TT').spectra()/lensed('TT').spectra()+nltt)
        return self.zeta.cf_from_cl(0, 1, cl_01)

    def zeta_0n1(self):
        cl_0n1 = np.zeros(lmax+1)
        for ell in range(0, lmax+1):
            cl_0n1 = (2*ell+1)/(4*np.pi)*np.sqrt(ell*(ell+1)) * \
                (unlensed('TT').spectra()/lensed('TT').spectra()+nltt)

        return self.zeta.cf_from_cl(0, -1, cl_0n1)

    def zeta_11(self):
        cl_11 = np.zeros(lmax+1)
        for ell in range(0, lmax+1):
            cl_11 = (2*ell+1)/(4*np.pi)*ell*(ell+1) * \
                (np.square(unlensed('TT').spectra())/(lensed('TT').spectra()+nltt))
        return self.zeta.cf_from_cl(1, 1, cl_11)

    def zeta_1n1(self):
        cl_1n1 = np.zeros(lmax+1)
        for ell in range(0, lmax+1):
            cl_1n1 = (2*ell+1)/(4*np.pi)*ell*(ell+1) * \
                (np.square(unlensed('TT').spectra())/(lensed('TT').spectra()+nltt))
        return self.zeta.cf_from_cl(1, -1, cl_1n1)

    def noise(self):
        ret = np.zeros(lmax+1)
        clL = self.zeta.cl_from_cf(lmax, -
                                   1, -1, self.zeta_00()*self.zeta_11() - self.zeta_01()*self.zeta_01()) + self.zeta.cl_from_cf(lmax, 1, -1, self.zeta_00()*self.zeta_1n1() - self.zeta_01()*self.zeta_0n1())

        for L in range(0, lmax+1):
            ret[L] = np.pi*L * (L+1)*clL(L)
        return 1./ret


if 0:
    plt.plot(ls, TT().noise())
    plt.show()

if 0:
    plt.plot(nltt)
    plt.show()

if 0:
    test = TT().zeta_00()
    print(test)
    plt.plot(test)
    plt.xscale('log')
    plt.ylabel('log')
    plt.show()

if 0:
    print(len(unlensed('TT').spectra()))
    print(unlensed('TT').spectra())
    print(type(bl))
    print(len(bl))

if 1:
    array = np.loadtxt(
        'planck_lensing_wp_highL_bestFit_20130627_scalCls.dat')
    print(array[0: (lmax-lmin+1), 1])
    print(len(array[0:(lmax-lmin+1), 1]))
