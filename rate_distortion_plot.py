import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from inference import *


sigma_2 = 0.1

filename = "pickle/all_eigvals.pkl"
with open(filename, 'rb') as handle:
	all_eighs = pickle.load(handle)

filename2 = "pickle/all_eigvals2.pkl"
with open(filename2, 'rb') as handle:
	all_eighs2 = pickle.load(handle)[2:]

print(len(all_eighs2))


all_distortions = []
for i in range(len(all_eighs)):
	eighs = all_eighs[i]
	distortions = []
	for k in range(1, len(eighs)):
		distortion1 = k*eighs[k]
		distortion2 = np.sum(eighs[k:])
		distortion = np.log2(1 + (distortion1 + distortion2)/sigma_2)
		distortion = distortion/np.log2(1 + 1/sigma_2)
		distortions.append(distortion)
	all_distortions.append(distortions)
for i in range(len(all_eighs2)):
	eighs = all_eighs2[i]
	distortions = []
	for k in range(1, len(eighs)):
		distortion1 = k*eighs[k]
		distortion2 = np.sum(eighs[k:])
		distortion = np.log2(1 + (distortion1 + distortion2)/sigma_2)
		distortion = distortion/np.log2(1 + 1/sigma_2)
		distortions.append(distortion)
	all_distortions.append(distortions)

all_rates = []
for i in range(len(all_eighs)):
	eighs = all_eighs[i]
	rates = []
	for k in range(1,len(eighs)):
		rate = 0.5*np.sum(np.log2(eighs[:k]/eighs[k]))
		rates.append(rate)
	all_rates.append(rates)
for i in range(len(all_eighs2)):
	eighs = all_eighs2[i]
	rates = []
	for k in range(1,len(eighs)):
		rate = 0.5*np.sum(np.log2(eighs[:k]/eighs[k]))
		rates.append(rate)
	all_rates.append(rates)

epsilon = 0.19
rate_distortions = []
errors = []
for i in range(len(all_distortions)):
	distortions = np.array(all_distortions[i])
	diffs = np.abs(distortions-epsilon)
	errors.append(np.min(diffs))
	opt_index = np.argmin(diffs)
	rate_distortions.append(all_rates[i][opt_index])
print(errors)

upper =313
plt.plot(np.arange(2, upper, 10), rate_distortions)
plt.xlabel('dimension')
plt.ylabel('rate distortion')
plt.title('Rate Distortion vs Dimension for epsilon = %s'%epsilon)
plt.show()

epsilon = 0.22
rate_distortions = []
for i in range(len(all_distortions)):
	distortions = np.array(all_distortions[i])
	diffs = np.abs(distortions-epsilon)
	opt_index = np.argmin(diffs)
	rate_distortions.append(all_rates[i][opt_index])

plt.plot(np.arange(2, upper, 10), rate_distortions)
plt.xlabel('dimension')
plt.ylabel('rate distortion')
plt.title('Rate Distortion vs Dimension for epsilon = %s'%epsilon)
plt.show()

epsilon = 0.18
rate_distortions = []
errors = []
for i in range(len(all_distortions)):
	distortions = np.array(all_distortions[i])
	diffs = np.abs(distortions-epsilon)
	opt_index = np.argmin(diffs)
	errors.append(np.min(diffs))
	rate_distortions.append(all_rates[i][opt_index])
print(errors)

plt.plot(np.arange(2, upper, 10), rate_distortions)
plt.xlabel('dimension')
plt.ylabel('rate distortion')
plt.title('Rate Distortion vs Dimension for epsilon = %s'%epsilon)
plt.show()

ds = np.arange(2, 503, 10)
plt.plot(all_distortions[0], all_rates[15], color='purple', label='d=%s'%(ds[15]))
plt.plot(all_distortions[2], all_rates[19], color='green', label='d=%s'%(ds[19]))
plt.plot(all_distortions[4], all_rates[23], color='blue', label='d=%s'%(ds[23]))
plt.plot(all_distortions[6], all_rates[27], color='red', label='d=%s'%(ds[27]))
plt.plot(all_distortions[20], all_rates[31], color='pink', label='d=%s'%(ds[31]))
plt.xlabel('epsilon')
plt.ylabel('Rate Distortion')
plt.legend()
plt.show()

