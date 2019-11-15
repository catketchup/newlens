import numpy as np
import matplotlib.pyplot as plt
import wignerd
import path
import pandas as pd
import scipy.interpolate
import load_data
import noise_model
import reconstruction_noise
import importlib

importlib.reload(reconstruction_noise)
importlib.reload(noise_model)
# calculate the EB estimator using lmy new result as a demo
lmax = 2000
lmin = 2
ls = np.arange(0, lmax + 1)


def t(ell):
    return (ell * (ell + 1.))


if 0:  # planck parameters
    bealm_fwhlm = 7
    nlev_t = 27  # telmperature noise level, in uk.arclmin
    nlev_p = np.sqrt(2) * 40

    Dl_TT_noise = t(ls) * reconstruction_noise.TT(
        lmin, lmax, bealm_fwhlm, nlev_t, nlev_p).noise() / (2 * np.pi)
    Dl_EB_noise = t(ls) * reconstruction_noise.EB(
        lmin, lmax, bealm_fwhlm, nlev_t, nlev_p).noise() / (2 * np.pi)

    plt.plot(ls[lmin:lmax], Dl_TT_noise[lmin:lmax])
    plt.plot(ls[lmin:lmax], Dl_EB_noise[lmin:lmax])
    plt.legend(['TT_noise', 'EB_noise'])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$L$')
    plt.ylabel(r'$[L[L+1] C_L^{\phi\phi} / 2\pi$')
    plt.show()

if 0:  # reference parameters
    bealm_fwhlm = 4
    nlev_t = 1  # telmperature noise level, in uk.arclmin
    nlev_p = np.sqrt(2)

    Dl_TT_noise = t(ls) * reconstruction_noise.TT(
        lmin, lmax, bealm_fwhlm, nlev_t, nlev_p).noise() / (2 * np.pi)
    Dl_EB_noise = t(ls) * reconstruction_noise.EB(
        lmin, lmax, bealm_fwhlm, nlev_t, nlev_p).noise() / (2 * np.pi)

    plt.plot(ls[lmin:lmax], Dl_TT_noise[lmin:lmax])
    plt.plot(ls[lmin:lmax], Dl_EB_noise[lmin:lmax])
    plt.legend(['TT_noise', 'EB_noise'])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$L$')
    plt.ylabel(r'$[L[L+1] C_L^{\phi\phi} / 2\pi$')
    plt.show()

if 1:  # plot Dl_TT_noise given a bunch of parameters
    bealm_fwhlm = [2, 4, 8, 16]
    nlev_t = 3  # telmperature noise level, in uk.arclmin
    nlev_p = 3
    for x in bealm_fwhlm:

        Dl_TT_noise = t(ls) * reconstruction_noise.TT(
            lmin, lmax, x, nlev_t, nlev_p).noise() / (2 * np.pi)

        plt.plot(ls[lmin:lmax], Dl_TT_noise[lmin:lmax])

    plt.legend(['2', '4', '8', '16'])
    plt.title('3uK-arcmin')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$L$')
    plt.ylabel(r'$[L[L+1] C_L^{\phi\phi} / 2\pi$')
    plt.show()

if 0:  # plot Dl_TT_noise given a bunch of parameters
    bealm_fwhlm = 4
    nlev_t = [1, 3, 10, 30]  # telmperature noise level, in uk.arclmin
    nlev_p = 3
    for x in nlev_t:

        Dl_TT_noise = t(ls) * reconstruction_noise.TT(
            lmin, lmax, bealm_fwhlm, x, nlev_p).noise() / (2 * np.pi)

        plt.plot(ls[lmin:lmax], Dl_TT_noise[lmin:lmax])

    plt.legend(['1', '3', '10', '30'])
    plt.title('4 beam')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$L$')
    plt.ylabel(r'$[L[L+1] C_L^{\phi\phi} / 2\pi$')
    plt.show()

if 0:  # plot Dl_EB_noise given a bunch of parameters
    bealm_fwhlm = [2, 4, 8, 16]
    nlev_t = 3  # telmperature noise level, in uk.arclmin
    nlev_p = np.sqrt(2) * nlev_t
    for x in bealm_fwhlm:

        Dl_TT_noise = t(ls) * reconstruction_noise.EB(
            lmin, lmax, x, nlev_t, nlev_p).noise() / (2 * np.pi)

        plt.plot(ls[lmin:lmax], Dl_TT_noise[lmin:lmax])

    plt.legend(['2', '4', '8', '16'])
    plt.title('3uK-arcmin')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$L$')
    plt.ylabel(r'$[L[L+1] C_L^{\phi\phi} / 2\pi$')
    plt.show()

    # filenalme_TT = './data/hu_TT.csv'
    # filenalme_EB = './data/hu_EB.csv'

    # Dl_TT_noise_hu = np.array(pd.read_csv(filenalme_TT))
    # Dl_EB_noise_hu = np.array(pd.read_csv(filenalme_EB))
