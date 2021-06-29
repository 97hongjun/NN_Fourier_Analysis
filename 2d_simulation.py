import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import time
from scipy.special import loggamma

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

X = np.arange(-1.0, 1.001, 0.02)
Y = np.arange(-1.0, 1.001, 0.02)
grid = []
for x in X:
  for y in Y:
    if x**2 + y**2 <= 1.0:
      grid.append([x, y])
grid = np.array(grid)

d = 2
t = len(grid)
print(t)
sigma_2 = 0.1
coeff = 0.3
print('Assembling Kernel Matrix')
Sigma = compute_covariance_matrix_massive(grid, 10,  coeff)
print('Computing Eigenvalues')
eig_vals, eig_vecs = np.linalg.eigh(Sigma/float(t))
print('Plotting')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

g_vec = []
for gridpt in grid:
  g_vec.append([np.sin(np.pi*gridpt[0]/2.0) + np.sin(np.pi*gridpt[1]/2.0)])
g_vec = np.array(g_vec)
g_vec = g_vec/np.linalg.norm(g_vec)


grid_axis = []
function = []
grid_axis2 = []
function2 = []

grid_axis3 = []
function3 = []
grid_axis4 = []
function4 = []

grid_axis5 = []
function5 = []
grid_axis6 = []
function6 = []

grid_axis7 = []
function7 = []
grid_axis8 = []
function8 = []

eig_index = 2

for k in range(len(grid)):
	if np.abs(grid[k][0]+grid[k][1]) < 0.01:
		grid_axis.append(grid[k])
		function.append(eig_vecs[:,-eig_index][k])
grid_axis = np.array(grid_axis)

for k in range(len(grid)):
	if np.abs(grid[k][0]-grid[k][1]) < 0.01:
		grid_axis2.append(grid[k])
		function2.append(eig_vecs[:,-eig_index][k])
grid_axis2 = np.array(grid_axis2)

for k in range(len(grid)):
	if np.abs(grid[k][0]) < 0.01:
		grid_axis3.append(grid[k])
		function3.append(eig_vecs[:,-eig_index][k])
grid_axis3 = np.array(grid_axis3)

for k in range(len(grid)):
	if np.abs(grid[k][1]) < 0.01:
		grid_axis4.append(grid[k])
		function4.append(eig_vecs[:,-eig_index][k])
grid_axis4 = np.array(grid_axis4)

for k in range(len(grid)):
	if np.abs(grid[k][0]-0.5) < 0.01:
		grid_axis5.append(grid[k])
		function5.append(eig_vecs[:,-eig_index][k])
grid_axis5 = np.array(grid_axis5)

for k in range(len(grid)):
	if np.abs(grid[k][0]+0.5) < 0.01:
		grid_axis6.append(grid[k])
		function6.append(eig_vecs[:,-eig_index][k])
grid_axis6 = np.array(grid_axis6)

for k in range(len(grid)):
	if np.abs(grid[k][1]-0.5) < 0.01:
		grid_axis7.append(grid[k])
		function7.append(eig_vecs[:,-eig_index][k])
grid_axis7 = np.array(grid_axis7)

for k in range(len(grid)):
	if np.abs(grid[k][1]+0.5) < 0.01:
		grid_axis8.append(grid[k])
		function8.append(eig_vecs[:,-eig_index][k])
grid_axis8 = np.array(grid_axis8)

# ax.scatter(grid.T[0][:int(len(grid.T[0])/2)], grid.T[1][:int(len(grid.T[0])/2)], eig_vecs[:,-2][:int(len(grid.T[0])/2)], alpha=0.05)

ax.scatter(grid_axis.T[0], grid_axis.T[1], function, alpha=0.1)
ax.scatter(grid_axis2.T[0], grid_axis2.T[1], function2, alpha=0.1)
ax.scatter(grid_axis3.T[0], grid_axis3.T[1], function3, alpha=0.1)
ax.scatter(grid_axis4.T[0], grid_axis4.T[1], function4, alpha=0.1)
ax.scatter(grid_axis5.T[0], grid_axis5.T[1], function5, alpha=0.1, color='green')
ax.scatter(grid_axis6.T[0], grid_axis6.T[1], function6, alpha=0.1, color='green')
ax.scatter(grid_axis7.T[0], grid_axis7.T[1], function7, alpha=0.1, color='red')
ax.scatter(grid_axis8.T[0], grid_axis8.T[1], function8, alpha=0.1, color='red')

ax.scatter(grid.T[0], grid.T[1], g_vec.T-eig_vecs[:,-2], alpha=0.1)

ax.set_xlabel('x')
ax.set_ylabel('y')
# ax.scatter(grid.T[0][:], grid.T[1][:], eig_vecs[:,-3][:])
plt.show()


plt.plot(np.log10(eig_vals[::-1]))
plt.show()