import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle
import time
from pathlib import Path
from inference import *


root = Path(".")
t = 1000
sigma_2 = 0.1

max_eighs = []
second_max_eighs = []
sample_complexities = []
sample_complexities2 = []
sample_complexities3 = []

path_eig = root / "pickle" / "max_eighs.pkl"
path_eig2 = root / "pickle" / "second_max_eighs.pkl"
path_sc1 = root / "pickle" / "sample_comp_1.pkl"
path_sc2 = root / "pickle" / "sample_comp_2.pkl"
path_sc3 = root / "pickle" / "sample_comp_3.pkl"

with open(path_eig, 'wb') as handle:
	pickle.dump(max_eighs, handle)

for d in range(2, 152, 4):
	sample_max_eighs = []
	sample_second_max_eighs = []
	sample_sample_complexities = []
	sample_sample_complexities2 = []
	sample_sample_complexities3 = []
	for sample in range(1):
		X = sample_ball(d, t+1)
		
		Sigma = compute_covariance_matrix_massive(X.T, 4)/float(t)
		s_time = time.time()
		eighs = np.linalg.eigvalsh(Sigma)
		e_time = time.time()
		print("Took %s Seconds to find Eigenvalues"%(e_time - s_time))
		eighs = eighs[::-1]
		max_eigh = eighs[0]
		second_max_eigh = eighs[1]
		sample_max_eighs.append(max_eigh)
		sample_second_max_eighs.append(second_max_eigh)
		first1 = True
		first2 = True
		for k in range(1, t):
			lhs = float(k)*eighs[k]/np.sum(eighs)
			rhs = np.sum(eighs[:k])/np.sum(eighs) - 0.95
			rhs2 = np.sum(eighs[:k])/np.sum(eighs) - 0.94
			rhs3 = np.sum(eighs[:k])/np.sum(eighs) - 0.93
			if lhs <= rhs and d > 10:
				sample_sample_complexities.append(k)
				break
			if lhs <= rhs2 and d > 10 and first1:
				sample_sample_complexities2.append(k)
				first1 = False
			if lhs <= rhs3 and d > 10 and first2:
				sample_sample_complexities3.append(k)
				first2 = False
		if k == t-1:
			sample_sample_complexities.append(k)
			if first1:
				sample_sample_complexities2.append(k)
			if first2:
				sample_sample_complexities3.append(k)
			print("didn't work: %s"%d, lhs, rhs)
	max_eighs.append(np.mean(sample_max_eighs))
	second_max_eighs.append(np.mean(sample_second_max_eighs))
	sample_complexities.append(np.mean(sample_sample_complexities))
	sample_complexities2.append(np.mean(sample_sample_complexities2))
	sample_complexities3.append(np.mean(sample_sample_complexities3))
	with open(path_eig, 'wb') as handle:
		pickle.dump(max_eighs, handle)
	with open(path_eig2, 'wb') as handle:
		pickle.dump(second_max_eighs, handle)
	with open(path_sc1, 'wb') as handle:
		pickle.dump(sample_complexities, handle)
	with open(path_sc2, 'wb') as handle:
		pickle.dump(sample_complexities2, handle)
	with open(path_sc3, 'wb') as handle:
		pickle.dump(sample_complexities3, handle)
	if d%10 == 0:
		print("finished d: %s"%d)
	