from healpy import sphtfunc
from pixell import curvedsky, enmap, enplot, utils
import numpy as np
import matplotlib

# shape, wcs = enmap.fullsky_geometry(res=20 * utils.arcmin, proj='car')
# print(shape)
# print(wcs)
# in_powspec = np.arange(2, 1000)
# mymap = curvedsky.rand_map(shape, wcs, in_powspec)
#enplot.show(enplot.plot(mymap))

## alm from a map
in_powspec = np.arange(2, 1000)
healpymap = sphtfunc.synfast(in_powspec, 2)
alm = sphtfunc.map2alm(healpymap)
#out_powspec = sphtfunc.anafast(healpymap)
m = np.array([0, 1])
index = sphtfunc.Alm.getidx(7, 1, m)
#print(index)
#print(alm[index])
#print(out_powspec)
print(alm)

lmin = 2
lmax = 3000


## test healpy.sphtfunc.almxfl
def fl1():
    ell = np.arange(lmin, lmax + 1)
    return ell * (ell + 1)


factor1 = sphtfunc.almxfl(alm, fl1())
print(factor1)

map1 = sphtfunc.alm2map(factor1, 2)
print(map1)


def fl2():
    ell = np.arange(lmin, lmax + 1)
    return (2 * ell + 1)**(1 / 2)


factor2 = sphtfunc.almxfl(alm, fl2())

map2 = sphtfunc.alm2map(factor2, 2)
print(map2)

map = map1 * map2
print(map)

phi_lm = sphtfunc.map2alm(map)
print(phi_lm)
