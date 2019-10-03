from healpy import sphtfunc
from pixell import curvedsky, enmap, enplot, utils
import numpy as np
import matplotlib

shape, wcs = enmap.fullsky_geometry(res=20 * utils.arcmin, proj='car')
print(shape)
print(wcs)
in_powspec = np.arange(2, 1000)
mymap = curvedsky.rand_map(shape, wcs, in_powspec)
#enplot.show(enplot.plot(mymap))

## alm from a map
healpymap = sphtfunc.synfast(test_powspec, 2)
alm = sphtfunc.map2alm(healpymap)
#out_powspec = sphtfunc.anafast(healpymap)
m = np.array([0, 1])
index = sphtfunc.Alm.getidx(7, 1, m)
print(index)
print(alm[index])
#print(out_powspec)
print(alm)
