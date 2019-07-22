import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import draft0
import cwignerd


def unlensed_ClEE():

    data = np.loadtxt(
        "planck_lensing_wp_highL_bestFit_20130627_scalCls.dat", usecols=(2), unpack=True)
    return data


x = unlensed_ClEE()
print(x)
plt.plot(x)
plt.show()
