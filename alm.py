from pixell import curvedsky, enmap, enplot, utils, powspec
import numpy as np
import matplotlib.pyplot as plt
import path
import healpy
shape, wcs = enmap.fullsky_geometry(res=20 * utils.arcmin, proj='car')
test_powspec = path.file(1, 'unlensed').file_data()
ps, _ = powspec.read_camb_scalar(test_powspec)

omap = curvedsky.rand_map(shape, wcs, ps)
enplot.show(enplot.plot(omap), 'mymap')
alm = curvedsky.map2alm(omap, lmax=200)
print(alm)
