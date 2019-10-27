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
lmax = 1000
nside = 400
# generate unlensed map using input unlensd ps
unlensed_ps = load_data.unlensed(lmin, lmax, 'TT').spectra()
unlensed_map = curvedsky.rand_map(shape, wcs, unlensed_ps)
#enplot.show(enplot.plot(unlensed_map))
cmb_alm = curvedsky.map2alm(unlensed_map, lmax=lmax)
print(cmb_alm)
# generate deflection potential using input
ls = np.arange(lmin, lmax)
input_cldd = load_data.input_cldd(lmin, lmax).spectra()
input_phi_cl = input_cldd[lmin:lmax] / (ls * (ls + 1))
phi_map = curvedsky.rand_map(shape, wcs, input_phi_cl)
#enplot.show(enplot.plot(phi_map))
phi_alm = curvedsky.map2alm(phi_map, lmax=lmax)

# generate lensed map using unlensed_alm and phi_alm
lensing.lens_map_curved(shape, wcs, phi_alm,
                        cmb_alm)  # this doesn't change cmb_alm

# lensed_map = curvedsky.alm2map(cmb_alm)
