# Implementing multivariate linear regression (also works for univariate)

# Reference - https://www.youtube.com/watch?v=qLCVj_Ej5VQ

import numpy as np

# Inference 
def predict(X, w):
  return X.dot(w)

# Training - Gradient descent
def compute_cost(X, y, w):
  n = X.shape[0]
  return np.linalg.norm(X.dot(w)-y)/n

def compute_gradient(X, y, w):
  n = X.shape[0]
  return (2*np.transpose(X).dot(X.dot(w)-y))/n

def gradient_descent(X, y, w, u, iters):
  cost = []
  for _ in range(iters):
    w = w - u*compute_gradient(X, y, w)
    cost.append(compute_cost(X, y, w))
  return w, cost

def fit(X, y, u, iters):
  m = X.shape[1]
  w0 = np.zeros(m)
  return gradient_descent(X, y, w0, u, iters)

# Training - Closed form
def closed_form(X, y):
  X_T = np.transpose(X)
  w = np.matmul(np.linalg.inv(np.matmul(X_T, X)), X_T).dot(y)
  return w