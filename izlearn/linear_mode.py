import numpy as np
import pandas as pd

class LinearRegression:
  def __init__(self, fit_intercept=True, copy_X=True, n_jobs=None, positive=False):
    self.coef_ = None
    self.intercept_ = None
    self.n_features_in_ = None
    self.feature_names_in_ = None
    self.rank_ = None
    self.singular_ = None
    self.fit_intercept_ = fit_intercept
    self.copy_X_ = copy_X
    self.n_jobs_ = n_jobs
    self.positive_ = positive

  def fit(self, X, y): 

    """
    Input:
    self: class
    X: array[float]
    y: array[float]

    Output:
    self: class
    """

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    if self.fit_intercept_:
      X_b = np.c_[np.ones((X.shape[0], 1)), X]
    else:
      X_b = X

    theta, _, rank, singular = np.linalg.lstsq(X_b, y, rcond=None)

    if self.fit_intercept_:
      self.intercept_ = theta[0]
      self.coef_ = theta[1:]
    else:
      self.intercept_ = 0.0
      self.coef_ = theta

    return self

  def predict(self, X):

    """
    Input: 
    self: class
    X: array[float]

    Output:
    y: [float, array[float]]
    """

    X = np.array(X, dtype=float)
    return X @ self.coef_ + self.intercept_

  def score(self, X, y): 

    """
    Output: 
    X: array[float]
    y: array[float]
    """
    y_pred = self.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot

  def get_params(self):

    """
    Input:
    self: class

    Output:
    copy_X: [True, False]
    fit_intercept: [True, False]
    n_jobs: [None, integer]
    positive: [True, False]
    """

    return {
        "copy_X": self.copy_X_,
        "fit_intercept": self.fit_intercept_,
        "n_jobs": self.n_jobs_,
        "positive": self.positive_
    }

  def set_params(self, **params): 
    
    """
    Input:
    self: class
    copy_X: [True, False]
    fit_intercept: [True, False]
    n_jobs: [None, integer]
    positive: [True, False]

    Output:
    self: class
    """

    if not params: return self
    valid = self.get_params().keys()

    for key, value in params.items():
      if key not in valid:
        raise ValueError(f"Param '{key}' not in valid. Key valid are {list(valid)}")
      setattr(self, key + "_", value)

    return self
