# http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/

import numpy as np

# Inference
def predict(X, W):
  n, m = X.shape
  m, k = W.shape

  P = np.exp(np.matmul(X, W))
  # Divide a row by its sum - https://stackoverflow.com/questions/16202348/numpy-divide-row-by-row-sum
  P = P/P.sum(axis=1)[:, None]

  return np.argmax(P, axis=1) 


# Training
def compute_cost(X, y, W):
  n, m = X.shape
  m, k = W.shape

  P = np.exp(np.matmul(X, W))
  P = P/P.sum(axis=1)[:, None]

  cost = 0
  for i in range(n):
    cost += np.log(P[i, int(y[i])])

  return cost


def compute_gradient(X, y, W):
  n, m = X.shape
  m, k = W.shape

  P = np.exp(np.matmul(X, W))
  P = P/P.sum(axis=1)[:, None]

  R = np.zeros((n, k))
  for i in range(n):
    R[i, int(y[i])] = 1

  return -1*np.matmul(np.transpose(X), R-P)


def gradient_descent(X, y, W, u, iters):
  cost = []
  for i in range(iters):
    W = W - u*compute_gradient(X, y, W)
    cost.append(compute_cost(X, y, W))
  return W, cost
  

def fit(X, y, u, iters):
  n, m = X.shape
  k = np.unique(y).shape[0]
  W0 = np.zeros((m, k))
  return gradient_descent(X, y, W0, u, iters)