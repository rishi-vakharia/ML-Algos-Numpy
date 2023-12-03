import numpy as np
import pandas as pd

def train_test_split(X, y, train_size=0.75):
  X_y = np.c_[X, y]   
  n = X_y.shape[0]
  n_train = int(n*train_size)
  np.random.shuffle(X_y)
  return X_y[:n_train, :-1], X_y[n_train:, :-1], X_y[:n_train, -1].flatten(), X_y[n_train:, -1].flatten()

# Implementing normalization and standardization for feature scaling
def normalize(x):
  min = x.min()
  max = x.max()
  return (x - min)/(max - min)

def standardize(x):
  mean = np.mean(x)
  std = np.mean(x)
  return (x - mean)/std

def mean_squared_error(y, yhat):
  return np.sum(np.square(y-yhat))/y.shape[0]

def mean_absolute_error(y, yhat):
  return np.sum(np.absolute(y-yhat))/y.shape[0]

def accuracy_score(y, yhat):
  return np.count_nonzero(yhat == y)/y.shape[0]

def f1_score(y, yhat):
  classes = np.unique(y)
  f1 = []
  for i in classes:
    TP = np.count_nonzero(np.logical_and(y==i, yhat==i))
    FP = np.count_nonzero(np.logical_and(y!=i, yhat==i))
    FN = np.count_nonzero(np.logical_and(y==i, yhat!=i))
    f1.append(2*TP/(2*TP+FP+FN))
  return np.array(f1).mean()

# Remove outliers from ith column of dataframe df
def remove_outlier(df, i):
  Q1 = np.percentile(df[i], 25)
  Q3 = np.percentile(df[i], 75)
  IQR = Q3 - Q1
  upper = np.where(df[i] >= (Q3+1.5*IQR))
  lower = np.where(df[i] <= (Q1-1.5*IQR))
  df.drop(upper[0], inplace = True)
  df.drop(lower[0], inplace = True)
  df.reset_index(inplace=True, drop=True)