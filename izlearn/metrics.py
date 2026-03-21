import numpy as np

def r2_score(y_true, y_pred):
  y_true = np.array(y_true, dtype=float)
  y_pred = np.array(y_pred, dtype=float)
  ss_res = np.sum((y_true - y_pred) ** 2)
  ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
  return 1 - ss_res / ss_tot

def mean_squared_error(y_true, y_pred):
  y_true = np.array(y_true, dtype=float)
  y_pred = np.array(y_pred, dtype=float)
  return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def mean_squared_log_error(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)

def median_absolute_error(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.median(np.abs(y_true - y_pred))

