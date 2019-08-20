import numpy as np
import matplotlib.pyplot as plt
import cwignerd
import wignerd
import path

# calculate the EB estimator using my new result as a demon
lmax = 3000
lmin = 2
ls = np.arange(0, lmax+1)
nlev_t = 5.  # temperature noise level, in uk.arcmin
nlev_p = 5.
beam_fwhm = 1.


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
data_version = 1


class unlensed:
    def __init__(self, estimator):
        self.estimator = estimator
        self.data = np.loadtxt(path.file(data_version, 'unlensed').file_data())

    def spectra(self):
        ell = np.arange(lmin, lmax+1)
        if self.estimator == 'TT':
            return np.concatenate([np.zeros(lmin), self.data[0: (lmax-lmin+1), 1]/(ell*(ell+1.))*2*np.pi])

        if self.estimator == 'EE':
            return np.concatenate([np.zeros(lmin), self.data[0: (lmax-lmin+1), 2]])


class lensed:
    def __init__(self, estimator):
        self.estimator = estimator
        self.data = np.loadtxt(path.file(data_version, 'lensed').file_data())

    def spectra(self):
        ell = np.arange(lmin, lmax+1)
        if self.estimator == 'TT':
            return np.concatenate([np.zeros(lmin), self.data[0: (lmax-lmin+1), 1]/(ell*(ell+1.))*2*np.pi])

        if self.estimator == 'EE':
            return np.concatenate([np.zeros(lmin), self.data[0: (lmax-lmin+1), 2]])

        if self.estimator == 'BB':
            return np.concatenate([np.zeros(lmin), self.data[0: (lmax-lmin+1), 3]])


class TT:
    def __init__(self):
        self.zeta = wignerd.gauss_legendre_quadrature(4501)
        # self.array1 = unlensed('TT').spectra()
        self.array1 = lensed('TT').spectra()  # to match quicklens
        self.array2 = lensed('TT').spectra()+nltt

    def zeta_00(self):
        cl_00 = np.zeros(lmax+1, dtype=complex)
        ell = np.arange(lmin, lmax+1)
        cl_00[ell] = (2*ell+1)/(4*np.pi)*(1./self.array2[ell])
        return self.zeta.cf_from_cl(0, 0, cl_00)

    def zeta_01(self):
        cl_01 = np.zeros(lmax+1, dtype=complex)
        ell = np.arange(lmin, lmax+1)
        cl_01[ell] = (2*ell+1)/(4*np.pi)*np.sqrt(ell*(ell+1)) * \
                     (self.array1[ell]/self.array2[ell])
        return self.zeta.cf_from_cl(0, 1, cl_01)

    def zeta_0n1(self):
        cl_0n1 = np.zeros(lmax+1, dtype=complex)
        ell = np.arange(lmin, lmax+1)
        cl_0n1[ell] = (2*ell+1)/(4*np.pi)*np.sqrt(ell*(ell+1)) * \
                      (self.array1[ell] / self.array2[ell])
        return self.zeta.cf_from_cl(0, -1, cl_0n1)

    def zeta_11(self):
        cl_11 = np.zeros(lmax+1, dtype=complex)
        ell = np.arange(lmin, lmax+1)
        cl_11[ell] = (2 * ell+1)/(4*np.pi)*ell*(ell+1) * \
                     (self.array1[ell]**2)/(self.array2[ell])
        return self.zeta.cf_from_cl(1, 1, cl_11)

    def zeta_1n1(self):
        cl_1n1 = np.zeros(lmax+1, dtype=complex)
        ell = np.arange(lmin, lmax+1)
        cl_1n1[ell] = (2*ell+1)/(4*np.pi)*ell*(ell+1) * \
            self.array1[ell]**2/(self.array2[ell])
        return self.zeta.cf_from_cl(1, -1, cl_1n1)

    def noise(self):
        ret = np.zeros(lmax+1, dtype=complex)
        clL = self.zeta.cl_from_cf(lmax, -1, -1, self.zeta_00()*self.zeta_11() - self.zeta_01()*self.zeta_01(
        )) + self.zeta.cl_from_cf(lmax, 1, -1, self.zeta_00()*self.zeta_1n1() - self.zeta_01()*self.zeta_0n1())
        ell = np.arange(lmin, lmax+1)

        ret[ell] = (np.pi*ell * (ell+1)*clL[ell])**-1
        return ret


class EE:
    def __init__(self):
        self.zeta = wignerd.gauss_legendre_quadrature(4501)
        self.array1 = unlensed('EE').spectra()
        self.array2 = lensed('EE').spectra()+nlbb
        self.array3 = lensed('BB').spectra()+nlbb


class EB:
    def __init__(self):
        self.zeta = wignerd.gauss_legendre_quadrature(4501)
        self.array1 = unlensed('EE').spectra()
#        self.array1 = lensed('EE').spectra()  # to match quicklens
        self.array2 = lensed('EE').spectra()+nlbb
        self.array3 = lensed('BB').spectra()+nlbb

    def zeta_33(self):
        cl_33 = np.zeros(lmax+1, dtype=complex)
        ell = np.arange(lmin, lmax+1)
        cl_33[ell] = (2*ell+1)/(4*np.pi) * (self.array1[ell]
                                            )**2/self.array2[ell]*(ell-2)*(ell+3)

        return self.zeta.cf_from_cl(3, 3, cl_33)

    def zeta_3n3(self):
        cl_3n3 = np.zeros(lmax+1, dtype=complex)
        ell = np.arange(lmin, lmax+1)
        cl_3n3[ell] = (2*ell+1)/(4*np.pi) * (self.array1[ell]
                                             )**2/self.array2[ell]*(ell-2)*(ell+3)

        return self.zeta.cf_from_cl(3, -3, cl_3n3)

    def zeta_31(self):
        cl_31 = np.zeros(lmax+1, dtype=np.complex)
        ell = np.arange(lmin, lmax+1)
        cl_31[ell] = (2*ell+1)/(4*np.pi) * \
            (self.array1[ell])**2/self.array2[ell] * \
            np.sqrt((ell-1)*(ell+2)*(ell-2)*(ell+3))

        return self.zeta.cf_from_cl(3, 1, cl_31)

    def zeta_3n1(self):
        cl_3n1 = np.zeros(lmax+1, dtype=np.complex)
        ell = np.arange(lmin, lmax+1)
        cl_3n1[ell] = (2*ell+1)/(4*np.pi) * \
            (self.array1[ell])**2/self.array2[ell] * \
            np.sqrt((ell-1)*(ell+2)*(ell-2)*(ell+3))

        return self.zeta.cf_from_cl(3, -1, cl_3n1)

    def zeta_11(self):
        cl_11 = np.zeros(lmax+1, dtype=complex)
        ell = np.arange(lmin, lmax+1)
        cl_11[ell] = (2*ell+1)/(4*np.pi) * self.array1[ell]**2 / \
            self.array2[ell]*(ell-1)*(ell+2)

        return self.zeta.cf_from_cl(1, 1, cl_11)

    def zeta_1n1(self):
        cl_1n1 = np.zeros(lmax+1, dtype=complex)
        ell = np.arange(lmin, lmax+1)
        cl_1n1[ell] = (2*ell+1)/(4*np.pi) * self.array1[ell]**2 / \
            self.array2[ell]*(ell-1)*(ell+2)

        return self.zeta.cf_from_cl(1, -1, cl_1n1)

    def zeta_22(self):
        cl_22 = np.zeros(lmax+1)
        ell = np.arange(lmin, lmax+1)
        cl_22[ell] = (2*ell+1)/(4*np.pi)*(1./self.array3[ell])
        return self.zeta.cf_from_cl(2, 2, cl_22)

    def zeta_2n2(self):
        cl_22 = np.zeros(lmax+1)
        ell = np.arange(lmin, lmax+1)
        cl_22[ell] = (2*ell+1)/(4*np.pi)*(1./self.array3[ell])
        return self.zeta.cf_from_cl(2, -2, cl_22)

    def noise(self):
        ret = np.zeros(lmax+1, dtype=np.complex)
        clL = self.zeta.cl_from_cf(lmax, 1, 1, self.zeta_33()*self.zeta_22() - 2*self.zeta_3n1()*self.zeta_2n2()+self.zeta_11(
        )*self.zeta_22()) - self.zeta.cl_from_cf(lmax, 1, -1, self.zeta_3n3()*self.zeta_2n2() - 2*self.zeta_31()*self.zeta_22()+self.zeta_1n1()*self.zeta_2n2())
        ell = np.arange(lmin, lmax+1)

        ret[ell] = ((np.pi/4.)*ell * (ell+1)*clL[ell])**-1
        return ret


def t(ell): return (ell*(ell+1.))


if 0:
    result1 = TT().noise()
    result2 = EB().noise()
    plt.plot(ls[1:3001], (t(ls)*result1)[1:3001])
    plt.plot(ls[1:3001], (t(ls)*result2)[1:3001])
    plt.show()
    plt.yscale('log')
    print('done')


if 0:
    result1 = TT().noise()
    result2 = EB().noise()
    plt.plot(ls[2:2000], (t(ls)*t(ls)*result1)[2:2000]/(2*np.pi))
    plt.plot(ls[2:2000], (t(ls)*t(ls)*result2)[2:2000]/(2*np.pi))
    plt.legend(['TT', 'EB'])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$L$')
    plt.ylabel(r'$[L[L+1] C_L^{\phi\phi} / 2\pi$')
    plt.show()
    print('done')

if 1:
    EB_nl = t(ls)*t(ls)*EB().noise()
    EB_ql = np.real(np.loadtxt("data/EB.dat", dtype=complex))
    plt.plot(ls[2:2000], EB_nl[2:2000]/(2*np.pi))
    plt.plot(ls[2:2000], EB_ql[2:2000])
    plt.legend(['EB_nl', 'EB_ql'])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$L$')
    plt.ylabel(r'$[L[L+1] C_L^{\phi\phi} / 2\pi$')
    plt.show()

if 0:
    TT_nl = t(ls)*t(ls)*TT().noise()
    TT_ql = np.real(np.loadtxt("data/TT.dat", dtype=complex))
    plt.plot(ls[2:2000], TT_nl[2:2000]/(2*np.pi))
    plt.plot(ls[2:2000], TT_ql[2:2000])
    plt.legend(['TT_ql', 'TT_nl'])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$L$')
    plt.ylabel(r'$[L[L+1] C_L^{\phi\phi} / 2\pi$')
    plt.show()
    print('done')

if 0:
    result1 = TT().noise()
    ql_data = np.real(np.loadtxt("data/TT.dat", dtype=complex))
    plt.plot(ls[2:2000], (t(ls)*t(ls)*result1)
             [2:2000]/(2*np.pi)/ql_data[2:2000])
    plt.legend('ratio')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$L$')
    plt.ylabel(r'quicklens / newlens')
    plt.show()
    print('done')
