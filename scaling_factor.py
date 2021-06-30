import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import time
from scipy.special import loggamma
import pickle
from pathlib import Path

def posterior_expectation(Xs, Ys, sigma_2):
  """
    Xs: array [X_0, X_1, ..., X_T]
    Ys: array [Y_1, Y_2, ..., Y_T]
    sigma_2: variance of noise
  """
  m, d = Xs.shape
  Sigma = compute_covariance_matrix(Xs, sigma_2)
  Sigma_inv = np.linalg.solve(Sigma, np.eye(m))
  coeff = -1.0/Sigma_inv[m-1, m-1]
  Sigma_col = Sigma_inv[m-1,:m-1]
  return coeff * np.dot(Sigma_col, Ys)

def posterior_variance(Xs, sigma_2):
  m, d = Xs.shape
  Sigma = compute_covariance_matrix(Xs, sigma_2)
  Sigma_inv = np.linalg.solve(Sigma, np.eye(m))
  return 1.0/Sigma_inv[m-1, m-1]

def compute_covariance_matrix(Xs, sigma_2):
  """
    Compute the covariance matrix from the kernel function.
    Xs: array [X_0, X_1, ..., X_T]
    sigma_2: variance of noise
  """
  m, d = Xs.shape
  t1 = np.reshape(np.tile(Xs, m), (m, m, d))
  t2 = np.reshape(np.tile(Xs, (m, 1)), (m, m, d))
  K1 = np.linalg.norm(t1 - t2, axis=2)
  coeff = 0.1
  Sigma = np.ones((m, m)) - coeff*K1
  return Sigma

def compute_covariance_matrix_massive(Xs, r, coeff):
  m, d = Xs.shape
  m_div = int(m/r)+1

  K = []
  for k in range(r):
    m1 = k*m_div
    m2 = min((k+1)*m_div, m)
    K_ks = []
    for j in range(r):
      n1 = j*m_div
      n2 = min((j+1)*m_div, m)
      if m1 == n1:
        K_kj = compute_diag(Xs, m1, m2)
      elif m1 < n1:
        K_kj = compute_upper(Xs, m1, m2, n1, n2)
      elif m1 > n1:
        K_kj = compute_lower(Xs, n1, n2, m1, m2)
      K_ks.append(K_kj)
    K_ks = np.hstack(K_ks)
    K.append(K_ks)
  K = np.vstack(K)
  return np.ones((m, m)) - coeff*K

def compute_diag(Xs, m1, m2):
  t1_11 = np.reshape(np.tile(Xs[m1:m2], m2-m1), (m2-m1, m2-m1, d))
  t2_11 = np.reshape(np.tile(Xs[m1:m2], (m2-m1, 1)), (m2-m1, m2-m1, d))
  return np.linalg.norm(t1_11 - t2_11, axis=2)

def compute_upper(Xs, m1, m2, n1, n2):
  t1_12 = np.reshape(np.tile(Xs[m1:m2], n2-n1), (m2-m1, n2-n1, d))
  t2_12 = np.reshape(np.tile(Xs[n1:n2], (m2-m1, 1)), (m2-m1, n2-n1, d))
  return np.linalg.norm(t1_12 - t2_12, axis=2)

def compute_lower(Xs, m1, m2, n1, n2):
  t1_21 = np.reshape(np.tile(Xs[n1:n2], m2-m1), (n2-n1, m2-m1, d))
  t2_21 = np.reshape(np.tile(Xs[m1:m2], (n2-n1, 1)), (n2-n1, m2-m1, d))
  return np.linalg.norm(t1_21 - t2_21, axis=2)

def sample_ball(d, n):
  X = np.random.normal(0.0, 1.0, (n, d))
  nrm = np.expand_dims(np.linalg.norm(X, axis=1), axis=1)
  X = np.divide(X, nrm)
  U = np.random.uniform(0, 1, n)**(1/float(d))
  U = np.expand_dims(U, axis=1)
  return np.multiply(X, U).T


def compute_covariance_matrix1d(Xs):
  """
    Compute the covariance matrix from the kernel function.
    Xs: array [X_0, X_1, ..., X_T]
    sigma_2: variance of noise
  """
  m, d = Xs.shape
  t1 = np.reshape(np.tile(Xs, m), (m, m, d))
  t2 = np.reshape(np.tile(Xs, (m, 1)), (m, m, d))
  K1 = np.abs(t1 - t2)
  K1 = np.reshape(K1, (m, m))
  coeff = 1.0
  Sigma = np.ones((m, m)) - coeff*K1
  return Sigma



ds = [2, 5, 10, 20, 50, 100, 200, 500]
t = 15000
coeffs = [0.1, 0.2, 0.3, 0.4]
root = Path(".")

for d in ds:
    for coeff in coeffs:
        np.random.seed(1)
        Xs = sample_ball(d, t)
        print('d: %s, coeff: %s'%(d, coeff))
        print('Assembling Kernel Matrix')
        s_time = time.time()
        Sigma = compute_covariance_matrix_massive(Xs.T, 50,  coeff)
        e_time = time.time()
        print('Took %s Seconds to Assemble Matrix'%(e_time - s_time))
        print('Computing Eigenvalues')
        s_time2 = time.time()
        eig_vals = np.linalg.eigvalsh(Sigma/float(t))
        e_time2 = time.time()
        filename = 'eig_vals_d%s_t%s_c%s.pkl'%(d, t, coeff)
        eigvals_filename = root / "pickle" / filename
        with open(eigvals_filename, 'wb') as handle:
            pickle.dump(eig_vals, handle)
        print('Took %s Seconds to Compute Eigenvalues'%(e_time2 - s_time2))