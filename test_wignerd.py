from sympy.physics.wigner import wigner_3j
import numpy as np


def factor(l1, L, l2):
    return -np.sqrt((L*(L+1)))/(l1*(l1+1)-l2*(l2+1)+L*(L+1))


def factor1(l1, s):
    return np.sqrt((l1-s)*(l1+s+1))


def factor2(l1, s):
    return np.sqrt((l1+s)*(l1-s+1))


def left(l1, L, l2, s):

    return wigner_3j(l1, L, l2, s, 0, -s)


def right(l1, L, l2, s):
    return factor(l1, L, l2)*(factor1(l1, s)*wigner_3j(l1, L, l2, s+1, -1, -s)+factor2(l1, s)*wigner_3j(l1, L, l2, s-1, 1, -s))


print(np.sqrt(105)*left(1, 2, 3, 0)-right(1, 2, 3, 0)*np.sqrt(35))


print(wigner_3j(1, 2, 2, 1, 0, -1))
print(wigner_3j(1, 2, 2, -1, 0, 1))
