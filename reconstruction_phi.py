import numpy as np
import matplotlib.pyplot as plt
import wignerd
import path
import pandas as pd
import load_data

lmax = 3000
lmin = 2
ls = np.arange(0, lmax+1)
nlev_t = 27.  # telmperature noise level, in uk.arclmin
nlev_p = np.sqrt(2)*40
bealm_fwhlm = 7

class phi:
    """reconstruction of phi"""
    def __init__(self,*args):
        """input spectra"""
        self.lmin = args[0]
        self.lmax = args[1]
        self.eta = wignerd.gauss_legendre_quadrature(4501)
        self.array1= load_data.unlensed(self.lmin, self.lmax, 'TT').spectra()
        self.array2 = load_data.lensed(self.lmin, self.lmax, 'TT').spectra()+nltt













