import numpy as np
import matplotlib.pyplot as plt


def bl(fwhm_arcmin, lmax):
    """ returns the map-level transfer function for a symmetric Gaussian beam.
         * fwhm_arcmin      = beam full-width-at-half-maximum (fwhm) in arcmin.
         * lmax             = maximum multipole.
    """
    ls = np.arange(0, lmax+1)
    return np.exp(-(fwhm_arcmin * np.pi/180./60.)**2 / (16.*np.log(2.)) * ls*(ls+1.))


lmax = 3000
ls = np.arange(0, lmax+1)
nlev_t = 5  # temperature noise level, in uk.arcmin
nlev_p = 5
beam_fwhm = 1

bl = bl(beam_fwhm, lmax)

# noise spectra
# bl is transfer functions
nltt = (np.pi/180./60.*nlev_t)**2 / bl**2
nlee = nlbb = (np.pi/180./60.*nlev_p)**2 / bl**2

print(nltt)
plt.plot(nltt)
plt.show()
