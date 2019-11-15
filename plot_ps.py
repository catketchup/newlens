import numpy as np
import matplotlib.pyplot as plt


def t(ls):
    return ls * (ls + 1)


def plot(lmin, lmax, ps, *args):
    ls = np.arange(0, lmax)

    for ips in ps:
        plt.plot(ls[lmin:lmax],
                 t(ls)[lmin:lmax] * ips[lmin:lmax] / (2 * np.pi))
    plt.legend(*args)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
