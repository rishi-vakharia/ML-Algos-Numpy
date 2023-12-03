# https://www.youtube.com/watch?v=3I8oX3OUL6I
# https://stats.stackexchange.com/questions/142505/how-to-use-naive-bayes-for-multi-class-problems
# https://stats.stackexchange.com/questions/21822/understanding-naive-bayes/21849#21849

import numpy as np
import scipy.stats as stats

# Training
def calculate_prior(y):
  classes, counts = np.unique(y, return_counts=True)
  prior = counts/counts.sum()
  return prior

def fit_guassians(X, y):
  classes = np.unique(y)
  n, m = X.shape
  mean = np.zeros((classes.shape[0], m))
  std = np.zeros((classes.shape[0], m))
  for i in classes:
    X_i = X[y==i]   
    mean[int(i)] = np.mean(X_i, axis=0)
    std[int(i)] = np.std(X_i, axis=0)
  return mean, std

def fit(X, y):
  prior = calculate_prior(y)
  mean, std = fit_guassians(X, y)
  theta = prior, mean, std
  return theta

# Inference
def calculate_likelihood(X, theta):
  prior, mean, std = theta
  c, m = mean.shape
  likelihood = np.zeros((X.shape[0], c))
  for i in range(c):
    l_i = 1
    for j in range(m):
      l_i *= stats.norm.pdf(X[:, j], mean[i, j], std[i, j])
    likelihood[:, i] = l_i
  return likelihood

def calculate_posterior(X, theta):
  prior, mean, std = theta
  posterior = calculate_likelihood(X, theta)
  for i in range(posterior.shape[0]):
    posterior[i] = posterior[i]*prior
  return posterior

def predict(X, theta):
  return np.argmax(calculate_posterior(X, theta), axis=1) 