import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('pickle/sample_comp_3.pkl', 'rb') as file:
	sample_complexities = pickle.load(file)

plt.plot(np.arange(14, 134, 4), sample_complexities[3:], color='green')
plt.show()