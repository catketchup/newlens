import numpy as np
l = 4
x = np.arange(l)
y = [0, 1, 2, 3]
y = np.array(y)

z = x * y
ell = np.arange(4)
k = np.zeros(4)
k[ell] = ell * y[
    ell]  # here ell can be either the ell array or the element in ell
print(z)
print(k)

a = np.arange(2, 4)
print(a)
