from healpy import sphtfunc, visufunc, fitsfunc
import importlib
import numpy as np
import matplotlib.pyplot as plt
import path
import load_data
import noise_model
import ipdb
import reconstruction_noise
from pixell import curvedsky, enmap, enplot, utils, lensing
import plot_ps
from astropy.io import fits

importlib.reload(plot_ps)

lmin = 2
lmax = 1000
ls = np.arange(0, lmax)
nside = 400


def t(ls):
    return ls * (ls + 1)


shape, wcs = enmap.fullsky_geometry(res=20 * utils.arcmin, proj='car')
# generate unlensed map using input unlensd ps
unlensed_ps = load_data.unlensed(lmin, lmax, 'TT').spectra()
unlensed_map = curvedsky.rand_map(shape, wcs, unlensed_ps)
unlensed_alm = curvedsky.map2alm(unlensed_map, lmax=lmax)

# generate deflection potential using input
input_cldd = load_data.input_cldd(lmin, lmax).spectra()

if 1:
    phi_lm = fitsfunc.read_alm('./fullskyPhi_alm_00000.fits')
    phi_lm = phi_lm.astype(np.cdouble)
    cl_phi = sphtfunc.alm2cl(a)[0:lmax]
    cl_dd = cl_phi * t(ls)

if 1:
    plot_ps.plot(lmin, lmax, [input_cldd, cl_dd], ['1', '2'])
