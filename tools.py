import numpy as np


def compute_diag(Xs, m1, m2):
	_, d = Xs.shape
	t1_11 = np.reshape(np.tile(Xs[m1:m2], m2-m1), (m2-m1, m2-m1, d))
	t2_11 = np.reshape(np.tile(Xs[m1:m2], (m2-m1, 1)), (m2-m1, m2-m1, d))
	return np.linalg.norm(t1_11 - t2_11, axis=2)

def compute_upper(Xs, m1, m2, n1, n2):
	_, d = Xs.shape
	t1_12 = np.reshape(np.tile(Xs[m1:m2], n2-n1), (m2-m1, n2-n1, d))
	t2_12 = np.reshape(np.tile(Xs[n1:n2], (m2-m1, 1)), (m2-m1, n2-n1, d))
	return np.linalg.norm(t1_12 - t2_12, axis=2)

def compute_lower(Xs, m1, m2, n1, n2):
	_, d = Xs.shape
	t1_21 = np.reshape(np.tile(Xs[n1:n2], m2-m1), (n2-n1, m2-m1, d))
	t2_21 = np.reshape(np.tile(Xs[m1:m2], (n2-n1, 1)), (n2-n1, m2-m1, d))
	return np.linalg.norm(t1_21 - t2_21, axis=2)