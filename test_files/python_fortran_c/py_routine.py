import numpy as np
import foo
import testfortran
import m
import cwignerd
foo.hello(15)
testfortran.addsub(4, 9)
print(m.m([1, 2, 3, 4, 5]))


class Call_Wignerd:
    def __init__(self, npoints):
        self.npoints = npoints
        self.zvec, self.wvec = cwignerd.init_gauss_legendre_quadrature(npoints)

    def cf_from_cl(self, s1, s2, cl):
        lmax = len(cl)-1
        return cwignerd.wignerd_cf_from_cl(s1, s2, 1, self.npoints, lmax, self.zvec, cl)


Value = Call_Wignerd(100)
s1 = 0
s2 = 0
cl = np.linspace(0, 1, 20)
print(Value.cf_from_cl(s1, s2, cl))
