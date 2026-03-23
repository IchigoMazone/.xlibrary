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


class SGDRegressors:
  def __init__(self, alpha=1e-4, average=False, early_stopping=False, epsilon=1e-1, eta0=1e-2, fit_intercept=True,
               l1_ratio=1.5e-1, learning_rate='invscaling',  loss='squared_error', max_iter=1000, n_iter_no_change=5,
               penalty='l2', power_t=2.5e-1,random_state=None, shuffle=True, tol=1e-3, validation_fraction=1e-1,
               verbose=0, warm_start=False):
    self.coef_ = None
    self.intercept_ = None
    self.n_iters_ = None
    self.t_ = None
    self.current_lr_ = None
    self.alpha_ = alpha
    self.average_ = average
    self.early_stopping_ = early_stopping
    self.epsilon_ = epsilon
    self.eta0_ = eta0
    self.fit_intercept_ = fit_intercept
    self.l1_ratio_ = l1_ratio
    self.learning_rate_ = learning_rate
    self.loss_ = loss
    self.max_iter_ = max_iter
    self.n_iter_no_change_ = n_iter_no_change
    self.penalty_ = penalty
    self.power_t_ = power_t
    self.random_state_ = random_state
    self.shuffle_ = shuffle
    self.tol_ = tol
    self.validation_fraction_ = validation_fraction
    self.verbose_ = verbose
    self.warm_start_ = warm_start

  def _compute_lr(self):
    if self.learning_rate_ == "constant": return self.eta0_
    elif self.learning_rate_ == "invscaling":
      return self.eta0_ / (self.t_ ** self.power_t_) if self.t_ > 0 else self.eta0_
    elif self.learning_rate_ == "adaptive": return self.current_lr_
    elif self.learning_rate_ == "optimal":
      typw = np.sqrt(1.0 / np.sqrt(self.alpha_))
      eta0 = typw / max(1.0, (1 + typw) * 0.5)
      lr = eta0 / (1.0 + eta0 * self.alpha_ * self.t_)
      return lr
    return self.eta0_

  def _apply_penalty(self, lr):
    if self.penalty_ == "l2": self.coef_ -= lr * self.alpha_ * self.coef_
    elif self.penalty_ == "l1": self.coef_ -= lr * self.alpha_ * np.sign(self.coef_)
    elif self.penalty_ == "elasticnet":
      self.coef_ -= lr * self.alpha_ * (
          self.l1_ratio_ * np.sign(self.coef_) +
          (1 - self.l1_ratio_) * self.coef_
      )

  def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n_samples, n_features = X.shape
    self.current_lr_ = self.eta0_

    if self.random_state_ is not None: np.random.seed(self.random_state_)

    if not self.warm_start_ or self.coef_ is None:
      self.coef_ = coef_init if coef_init is not None else np.zeros(n_features)
      self.intercept_ = intercept_init if intercept_init is not None else np.zeros(1)
      self.n_iter_ = 0
      self.t_ = 0

    if sample_weight is None: sample_weight = np.ones(n_samples)
    else:
      sample_weight = np.array(sample_weight, dtype=float)
      if sample_weight.shape[0] != n_samples:
        raise ValueError(f'Sample_weight {sample_weight.shape[0]} not X {n_samples}')
      if np.any(sample_weight < 0):
        raise ValueError(f'All value sample > 0')

    if self.early_stopping_:
      n_val = int(n_samples * self.validation_fraction_)
      X_val, y_val = X[:n_val], y[:n_val]
      X, y = X[n_val:], y[n_val:]
      sample_weight = sample_weight[n_val:]
      n_samples = X.shape[0]

    no_improve_count = 0
    best_loss = float('inf')

    for epoch in range(self.max_iter_):
      if self.shuffle_:
        idx = np.random.permutation(n_samples)
        X, y, sample_weight = X[idx], y[idx], sample_weight[idx]

      prev_coef = self.coef_.copy()
      for i in range(n_samples):
        lr = self._compute_lr()
        y_pred = X[i] @ self.coef_ + self.intercept_

        if np.any(np.isnan(self.coef_)):
          print(f"NaN tại epoch {epoch}, sample {i}")
          return self

        error = y_pred - y[i]

        dw = sample_weight[i] * error * X[i]
        db = sample_weight[i] * error

        self.coef_ -= lr * dw
        if self.fit_intercept_:
          self.intercept_ -= lr * db

        self._apply_penalty(lr)
        self.t_ += 1

      self.n_iter_ += 1

      if self.early_stopping_:
        val_loss = np.mean((X_val @ self.coef_ + self.intercept_ - y_val) ** 2)
        if val_loss < best_loss - self.tol_:
          best_loss = val_loss
          no_improve_count = 0
        else:
          no_improve_count += 1

        if no_improve_count >= self.n_iter_no_change_:
          if self.learning_rate_ == "adaptive":
            self.current_lr_ /= 5
            no_improve_count = 0

          if self.verbose_:
            print(f'Early_stopping: {epoch}')
          break
      else:
        if np.max(np.abs(self.coef_ - prev_coef)) < self.tol_:
          break

      if self.verbose_:
        loss = np.mean((X @ self.coef_ + self.intercept_ - y) ** 2)
        print(f"Epoch {epoch + 1}: loss = {loss:.6f}")

    return self

  def partial_fit(self, X, y, sample_weight=None):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n_samples, n_features = X.shape

    if self.coef_ is None:
      self.coef_ = np.zeros(n_features)
      self.intercept_ = np.zeros(1)
      self.t_ = 0
      self.current_lr = self.eta0_

    if sample_weight is None: sample_weight = np.ones(n_samples)
    else:
      sample_weight = np.array(sample_weight, dtype=float)
      if sample_weight.shape[0] != n_samples:
        raise ValueError(f'Sample_weight {sample_weight.shape[0]} not X {n_samples}')
      if np.any(sample_weight < 0):
        raise ValueError(f'All value sample > 0')

    for i in range(n_samples):
      lr = self._compute_lr()
      y_pred = X[i] @ self.coef_ + self.intercept_
      error = y_pred - y[i]

      self.coef_ -= lr * sample_weight[i] * error * X[i]
      if self.fit_intercept_:
        self.intercept_ -= lr * sample_weight[i] * error

      self._apply_penalty(lr)
      self.t_ += 1

    return self

  def predict(self, X):
    X = np.array(X, dtype=float)
    return X @ self.coef_ + self.intercept_

  def score(self, X, y, sample_weight=None):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    y_pred = self.predict(X)
    ss_res = np.sum((y_pred - y) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot

  def get_params(self, deep=True):
    return {
        "alpha": self.alpha_,
        "average": self.average_,
        "early_stopping": self.early_stopping_,
        "epsilon": self.epsilon_,
        "eta0": self.eta0_,
        "fit_intercept": self.fit_intercept_,
        "l1_ratio": self.l1_ratio_,
        "learning_rate": self.learning_rate_,
        "loss": self.loss_,
        "max_iter": self.max_iter_,
        "n_iter_no_change": self.n_iter_no_change_,
        "penalty": self.penalty_,
        "power_t": self.power_t_,
        "random_state": self.random_state_,
        "shuffle": self.shuffle_,
        "tol": self.tol_,
        "validation_fraction": self.validation_fraction_,
        "verbose": self.verbose_,
        "warm_start": self.warm_start_
    }

  def set_params(self, **params):
    if not params: return self

    valid = self.get_params().keys()
    for key, value in params.items():
      if key not in valid:
        raise ValueError(f"Param '{key}' not in valid. Key valid are {list(valid)}")
      setattr(self, key + "_", value)

    return self


