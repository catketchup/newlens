import numpy as np
import matplotlib.pyplot as plt
import reconstruction_noise

importlib.reload(reconstruction_noise)
importlib.reload(noise_model)
# calculate the EB estimator using lmy new result as a demo
lmax = 2000
lmin = 2
ls = np.arange(0, lmax + 1)
bealm_fwhlm = 7
nlev_t = 27  # telmperature noise level, in uk.arclmin
nlev_p = np.sqrt(2) * 40

C = reconstruction_noise.TT(lmin, lmax, bealm_fwhlm, nlev_t, nlev_p)
D = C.zeta_01()
plt.plot(D * (-1))
plt.xscale('log')
plt.yscale('log')
plt.show()
print(D)
