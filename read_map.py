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

shape, wcs = enmap.fullsky_geometry(res=20 * utils.arcmin, proj='car')

lmin = 2
lmax = 1000
ls = np.arange(0, lmax)
nside = 400


def t(ls):
    return ls * (ls + 1)


def lensed_lm():
    alm = fitsfunc.read_alm('./fullskyLensedCMB_alm_set00_00000.fits')
    alm = alm.astype(np.cdouble)
    return alm


def obs_lm_TT():
    alm = fitsfunc.read_alm('./simu_map.py')
    alm = alm.astype(np.cdouble)
    return alm


def lensed_cl():
    cl = sphtfunc.alm2cl(lensed_alm())
    return cl
