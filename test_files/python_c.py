import numpy as np
import ctypes
adder = ctypes.cdll.LoadLibrary('./adder.so')

a = ctypes.c_float(5.5)
b = ctypes.c_float(4.1)

add_float = adder.add_float
add_float.restype = ctypes.c_float
print("5.5+4.1="+str(add_float(a, b)))

add_int = adder.add_int
print("4+5="+str(add_int(4, 5)))

cwignerd = ctypes.cdll.LoadLibrary('./cwignerd.so')
npoints = 100
zvec = np.linspace(-1, 1, 20)
s1 = 0
s2 = 0
lmax = 100
cl = 1
return (cwignerd.wignerd_cf_from_cl(s1, s2, 1, npoints, lmax, zvec, cl))
