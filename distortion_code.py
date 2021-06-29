import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from inference import *


d = 100
t = 10000

filename = "pickle/distortion_d%s_t%s.pkl"%(d, t)
with open(filename, 'wb') as handle:
	pickle.dump([], handle)
X = sample_ball(d,t)
Sigma = compute_covariance_matrix_massive(X.T, 20)/float(t)
s_time = time.time()
eighs = np.linalg.eigvalsh(Sigma)[::-1]
e_time = time.time()
print("Took %s Seconds to find Eigenvalues"%(e_time - s_time))
with open(filename, 'wb') as handle:
	pickle.dump(eighs, handle)

plt.plot(np.arange(0, len(eighs), 1), np.log10(eighs), color='blue')
plt.show()

distortions = []
for k in range(1, len(eighs)):
	distortion1 = k*eighs[k]
	distortion2 = np.sum(eighs[k:])
	distortions.append(distortion1 + distortion2)

plt.plot(np.arange(1, len(eighs), 1), distortions, color='blue')
plt.show()