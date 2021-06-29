import numpy as np


def sample_neural_network(n, d, X):
  W_2 = np.random.normal(0.0, 1.0/np.sqrt(n), n)
  W_1 = np.random.normal(0.0, 1.0, (n, d))
  nrm = np.expand_dims(np.sqrt(np.sum(np.square(W_1), axis=1)), axis=1)
  W_1 = np.divide(W_1, nrm)
  B_1 = np.expand_dims(np.random.uniform(-1, 1, n), axis=1)
  
  H_1 = np.sign(np.dot(W_1, X) + B_1)
  return np.dot(W_2, H_1)