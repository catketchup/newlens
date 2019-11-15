import numpy as np

lmin = 2
lmax = 3000
tfname = '../planck_lensing_wp_highL_bestFit_20130627_scalCls.dat'
tarray = np.loadtxt(tfname)

newarray = np.concatenate([np.zeros(lmin), tarray[0:(lmax-lmin+1), 1]])

print(newarray)
