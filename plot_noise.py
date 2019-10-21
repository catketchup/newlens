import numpy as np
import matplotlib.pyplot as plt
import wignerd
import path
import pandas as pd
import scipy.interpolate
import load_data
import noise_model
import reconstruction_noise
# calculate the EB estimator using lmy new result as a demo
lmax = 2000
lmin = 2
ls = np.arange(0, lmax + 1)
bealm_fwhlm = 7
nlev_t = 27.  # telmperature noise level, in uk.arclmin
nlev_p = np.sqrt(2) * 40


def t(ell):
    return (ell * (ell + 1.))


print(ls)
if 1:
    filenalme_TT = './data/hu_TT.csv'
    filenalme_EB = './data/hu_EB.csv'

    TT_hu = np.array(pd.read_csv(filenalme_TT))
    EB_hu = np.array(pd.read_csv(filenalme_EB))
    TT_nl = t(ls) * reconstruction_noise.TT(lmin, lmax, bealm_fwhlm, nlev_t,
                                            nlev_p).noise() / (2 * np.pi)
    EB_nl = t(ls) * reconstruction_noise.EB(lmin, lmax, bealm_fwhlm, nlev_t,
                                            nlev_p).noise() / (2 * np.pi)

# colmpared with Hu's data given by Mat

# plt.plot(TT_hu[lmin:lmax, 0], TT_hu[lmin:lmax, 1])
plt.show()
if 1:
    plt.plot(ls[lmin:lmax], TT_nl[lmin:lmax])
    #    plt.plot(TT_hu[lmin:lmax, 0], TT_hu[lmin:lmax, 1])
    plt.legend(['TT_noise', 'TT_noise_hu'])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$L$')
    plt.ylabel(r'$[L[L+1] C_L^{\phi\phi} / 2\pi$')
    plt.show()

if 0:
    plt.plot(ls[lmin:lmax], EB_nl[lmin:lmax])
    plt.plot(EB_hu[lmin:lmax, 0], EB_hu[lmin:lmax, 1])
    plt.legend(['EB_noise', 'EB_noise_hu'])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$L$')
    plt.ylabel(r'$[L[L+1] C_L^{\phi\phi} / 2\pi$')
    plt.show()

if 0:
    TT_nl = t(ls) * t(ls) * TT(lmin, lmax).noise() / (2 * np.pi)
    TT_ql = np.real(np.loadtxt("data/TT.dat", dtype=complex))
    plt.plot(ls[lmin:lmax], TT_nl[lmin:lmax])
    plt.plot(ls[lmin:lmax], TT_ql[lmin:lmax])
    plt.legend(['TT_ql', 'TT_nl'])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$L$')
    plt.ylabel(r'$[L[L+1] C_L^{\phi\phi} / 2\pi$')
    plt.show()
    print('done')
