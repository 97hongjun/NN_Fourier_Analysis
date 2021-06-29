import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from inference import *


t = 10000
filename1 = 'pickle/all_eigvals2.pkl'
filename2 = 'pickle/dom_eigvals2.pkl'
all_eighs = []
dom_eighs = []
for d in range(142, 503, 10):
	np.random.seed(1)
	X = sample_ball(d,t)
	Sigma = compute_covariance_matrix_massive(X.T, 10)/float(t)
	s_time = time.time()
	eighs = np.linalg.eigvalsh(Sigma)[::-1]
	e_time = time.time()
	all_eighs.append(eighs)
	dom_eighs.append(np.sum(eighs[:d+1]))
	print("Took %s Seconds to find Eigenvalues for dimension %s"%(e_time - s_time, d))
	with open(filename1, 'wb') as handle:
		pickle.dump(all_eighs, handle)
	with open(filename2, 'wb') as handle:
		pickle.dump(dom_eighs, handle)

plt.plot(np.arange(2, 503, 10), dom_eighs, color='purple')
plt.show()