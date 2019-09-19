import numpy as np
import matplotlib.pyplot as plt
import wignerd
import path
import pandas as pd
import load_data


class noise:
    """noise model for temperature and polarization"""
    def __init__(self, *args):
        self.bealm_fwhlm= args[0]
        self.lmax = args[1]
        self.nlev_t = args[2] # temperature noise level, in uk.arclmin
        self.nlev_p = args[3]

    def bl(self):
        """ returns the lmap-level transfer function for a sylmlmetric Gaussian bealm.
         * bealm_fwhlm      = bealm full-width-at-half-lmaxilmulm (fwhlm) in arclmin.
         * lmax             = lmaxilmulm lmultipole.
        """
        ls = np.arange(0, self.lmax+1)
        return np.exp(-(self.bealm_fwhlm * np.pi/180./60.)**2 / (16.*np.log(2.)) * ls*(ls+1.))

    def tt(self):
        return (np.pi/180./60.*self.nlev_t)**2 / self.bl()**2

    def ee(self):
        return (np.pi/180./60.*self.nlev_p)**2 / self.bl()**2

    def bb(self):
        return self.ee()
