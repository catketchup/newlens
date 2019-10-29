from healpy import sphtfunc, visufunc
import numpy as np
import matplotlib.pyplot as plt
import path
import load_data
import noise_model
import ipdb
import reconstruction_noise
from pixell import curvedsky, enmap, enplot, utils, lensing

shape, wcs = enmap.fullsky_geometry(res=20 * utils.arcmin, proj='car')

print(shape)
lmin = 2
lmax = 2000
nside = 400
# generate unlensed map using input unlensd ps
unlensed_ps = load_data.unlensed(lmin, lmax, 'TT').spectra()
unlensed_map = curvedsky.rand_map(shape, wcs, unlensed_ps)
unlensed_alm = curvedsky.map2alm(unlensed_map, lmax=lmax)

# generate deflection potential using input
ls = np.arange(lmin, lmax)
input_cldd = load_data.input_cldd(lmin, lmax).spectra()
input_phi_cl = input_cldd[lmin:lmax] / (ls * (ls + 1))
phi_map = curvedsky.rand_map(shape, wcs, input_phi_cl)
phi_alm = curvedsky.map2alm(phi_map, lmax=lmax)

# generate lensed map using unlensed_alm and phi_alm
lensed_map = lensing.lens_map_curved(
    shape, wcs, phi_alm, unlensed_alm, output="l")[0]
lensed_alm = curvedsky.map2alm(lensed_map, lmax=lmax)
lensed_ps = sphtfunc.alm2cl(lensed_alm)
ls = np.arange(0, lmax)


def t(ls):
    return ls * (ls + 1)


if 0:  # plot plots
    enplot.show(enplot.plot(phi_map))
    enplot.show(enplot.plot(unlensed_map))
    enplot.show(enplot.plot(lensed_map))

if 1:  # plot powerspectra
    plt.plot(ls[lmin:lmax],
             t(ls)[lmin:lmax] * unlensed_ps[lmin:lmax] / (2 * np.pi))
    plt.plot(ls[lmin:lmax],
             t(ls)[lmin:lmax] * lensed_ps[lmin:lmax] / (2 * np.pi))

    plt.xscale('log')
    plt.yscale('log')
    plt.show()

if 0:
    plt.plot(ls[lmin:lmax], t(ls)[lmin:lmax] * phi_alm[lmin:lmax])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

if 0:
    enplot.show(enplot.plot(unlensed_map))
    enplot.show(enplot.plot(phi_map))

if 0:
    enplot.show(enplot.plot(lensed_map))
